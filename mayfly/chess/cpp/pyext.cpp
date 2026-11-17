// mayfly_cppsearch, the C++ tree (search2.h) as a Python extension.
//
//   t = mayfly_cppsearch.Tree(settings_dict, evaluate, tb=None, ts=None, device="cuda", graph=True)
//       ts is an export_ts.py trace. The network then runs in C++ (LibTorch build) and evaluate is unused.
//   evaluate(B, wide, ids, cast, hm, ep, I, M, widths) -> (pol, val, qv), each bytes-like float32
//       ids int64 B*64, cast float32 B*4, hm int64 B, ep int64 B, I int64 B*wide, M uint8 B*wide, widths int32 B
//       pol B*wide (logits cut to each row's legal moves), val B, qv B*wide
//   t.reset(); carried = t.begin(fen, [uci, ...], reuse); t.discount(d, q0); t.needs_eval(); t.expand_root()
//   code = t.step(target, deadline)      0 batch, 1 nothing gathered, 2 saturated. deadline is time.time() or -1
//   t.choose() -> index or -1; t.root() -> dict; t.pv(first, depth) -> [uci]; t.tree_nodes(); t.stats() -> dict
//   With settings "pipeline" (MCTS's key, 0 off), t.finish() backs up the batch still on the card, t.held() gives
//   its held collisions and t.in_flight() says whether one is out. With settings "_prof" 1, t.prof() gives a dict
//   of seconds by part of a step, and the network's own timers (the CUDA net's milliseconds).
//
// step() lets go of the GIL while it walks the tree and takes it back only for the network callback, so a ponder
// running in a thread leaves the engine's main thread free between batches, as the Python search does.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <memory>
#include <string>
#include <vector>

#include "search2.h"
#include "torchnet.h"
#ifdef MAYFLY_CUDANET
#include "../cudanet/cudanet_backend.h"
#endif

using namespace mayfly2;

namespace {

struct PyError : std::exception {};   // a Python exception is set (or saved) and must reach the caller

class PyNet : public Net {
public:
    PyNet(PyObject* fn, PyObject* tbfn, Net* inner) : fn_(fn), tb_(tbfn), inner_(inner)
    {
        Py_XINCREF(fn_);
        Py_XINCREF(tb_);
    }
    ~PyNet() override
    {
        delete inner_;
        Py_XDECREF(fn_);
        Py_XDECREF(tb_);
    }
    PyObject* saved_type = nullptr;
    PyObject* saved_value = nullptr;
    PyObject* saved_tb = nullptr;

    void evaluate(const Batch& in, Out& out) override
    {
        if (inner_) {               // the network in C++ (LibTorch), no Python and no GIL
            inner_->evaluate(in, out);
            return;
        }
        call(in, out);
    }

    // the pipeline's halves, the C++ network's own or the callback run at collect
    void submit(const Batch& in) override
    {
        if (inner_) inner_->submit(in);
        else Net::submit(in);
    }
    void collect(Out& out) override
    {
        if (inner_) {
            inner_->collect(out);
            return;
        }
        if (queued_.empty()) throw std::runtime_error("collect with no batch queued");
        Batch b = std::move(queued_.front());
        queued_.pop_front();
        call(b, out);
    }
    void timers_on(bool on) override
    {
        if (inner_) inner_->timers_on(on);
    }
    std::vector<std::pair<std::string, double>> timers() override
    {
        return inner_ ? inner_->timers() : std::vector<std::pair<std::string, double>>();
    }

    // the Python callback, under the GIL
    void call(const Batch& in, Out& out)
    {
        PyGILState_STATE g = PyGILState_Ensure();
        bool ok = false;
        PyObject* r = PyObject_CallFunction(
            fn_, "iiy#y#y#y#y#y#y#", in.B, in.wide,
            (const char*)in.ids.data(), (Py_ssize_t)(in.ids.size() * sizeof(int64_t)),
            (const char*)in.cast.data(), (Py_ssize_t)(in.cast.size() * sizeof(float)),
            (const char*)in.hm.data(), (Py_ssize_t)(in.hm.size() * sizeof(int64_t)),
            (const char*)in.ep.data(), (Py_ssize_t)(in.ep.size() * sizeof(int64_t)),
            (const char*)in.I.data(), (Py_ssize_t)(in.I.size() * sizeof(int64_t)),
            (const char*)in.M.data(), (Py_ssize_t)in.M.size(),
            (const char*)in.widths.data(), (Py_ssize_t)(in.widths.size() * sizeof(int)));
        if (r && PyTuple_Check(r) && PyTuple_GET_SIZE(r) == 3) {
            ok = take(PyTuple_GET_ITEM(r, 0), out.pol) && take(PyTuple_GET_ITEM(r, 1), out.val)
                 && take(PyTuple_GET_ITEM(r, 2), out.qv);
        } else if (r) {
            PyErr_SetString(PyExc_TypeError, "evaluate must return (pol, val, qv)");
        }
        Py_XDECREF(r);
        if (!ok) {
            PyErr_Fetch(&saved_type, &saved_value, &saved_tb);   // carried to the caller after the tree is mended
            PyGILState_Release(g);
            throw PyError();
        }
        PyGILState_Release(g);
    }

    // MCTS._tb_value through the Python callable tb(fen) -> None or float. A failure there is a miss, as
    // _tb_value itself treats a failed probe.
    bool tb(const std::string& fen, double& value) override
    {
        if (!tb_) return false;
        PyGILState_STATE g = PyGILState_Ensure();
        bool hit = false;
        PyObject* r = PyObject_CallFunction(tb_, "s", fen.c_str());
        if (r && r != Py_None) {
            value = PyFloat_AsDouble(r);
            hit = !PyErr_Occurred();
        }
        if (PyErr_Occurred()) PyErr_Clear();
        Py_XDECREF(r);
        PyGILState_Release(g);
        return hit;
    }

private:
    static bool take(PyObject* o, std::vector<float>& dst)
    {
        Py_buffer v;
        if (PyObject_GetBuffer(o, &v, PyBUF_SIMPLE) != 0) return false;
        const size_t n = (size_t)v.len / sizeof(float);
        dst.assign((const float*)v.buf, (const float*)v.buf + n);
        PyBuffer_Release(&v);
        return true;
    }
    PyObject* fn_;
    PyObject* tb_;
    Net* inner_;
};

struct TreeObject {
    PyObject_HEAD
    PyNet* net;
    Tree* tree;
};

void restore(TreeObject* self)
{
    if (self->net->saved_type) {
        PyErr_Restore(self->net->saved_type, self->net->saved_value, self->net->saved_tb);
        self->net->saved_type = self->net->saved_value = self->net->saved_tb = nullptr;
    } else if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "the C++ search failed");
    }
}

double num(PyObject* d, const char* k, double dflt)
{
    PyObject* v = PyDict_GetItemString(d, k);   // borrowed
    if (!v || v == Py_None) return dflt;
    double x = PyFloat_AsDouble(v);
    if (PyErr_Occurred()) { PyErr_Clear(); return dflt; }
    return x;
}

PyObject* bytes_of(const std::vector<float>& v)
{
    return PyBytes_FromStringAndSize((const char*)v.data(), (Py_ssize_t)(v.size() * sizeof(float)));
}

int Tree_init(TreeObject* self, PyObject* args, PyObject*)
{
    PyObject *d, *fn, *tbfn = Py_None, *tsobj = Py_None;
    const char* device = "cuda";
    int graph = 1;
    if (!PyArg_ParseTuple(args, "O!O|OOsp", &PyDict_Type, &d, &fn, &tbfn, &tsobj, &device, &graph)) return -1;
    if (tsobj == Py_None && !PyCallable_Check(fn)) {
        PyErr_SetString(PyExc_TypeError, "evaluate must be callable when there is no trace");
        return -1;
    }
    Net* inner = nullptr;
    if (tsobj != Py_None) {
        const char* ts = PyUnicode_AsUTF8(tsobj);
        if (!ts) return -1;
        std::string tsp(ts);
        // The hand-tuned CUDA net (chess/cudanet) takes a .cnw weight file in the trace's place. Its numbers are
        // not the LibTorch trace's, so only a version sealed for it may pass one.
        if (tsp.size() > 4 && tsp.compare(tsp.size() - 4, 4, ".cnw") == 0) {
#ifdef MAYFLY_CUDANET
            try {
                int dev = 0;
                std::string ds(device);
                if (ds.rfind("cuda:", 0) == 0) dev = std::atoi(ds.c_str() + 5);
                inner = make_cuda_net(tsp, dev);
            } catch (const std::exception& e) {
                PyErr_Format(PyExc_RuntimeError, "the CUDA net did not load: %s", e.what());
                return -1;
            }
#else
            PyErr_SetString(PyExc_RuntimeError, "this build has no CUDA net (build with MAYFLY_CUDANET)");
            return -1;
#endif
        } else {
#ifdef MAYFLY_TORCH
        try {
            inner = make_torch_net(ts, device, graph != 0);
        } catch (const std::exception& e) {
            PyErr_Format(PyExc_RuntimeError, "the trace did not load: %s", e.what());
            return -1;
        }
#else
        (void)ts;
        PyErr_SetString(PyExc_RuntimeError, "this build has no LibTorch (build_rig.bat builds one with it)");
        return -1;
#endif
        }
    }
    Settings s;
    s.batch = (int)num(d, "batch", s.batch);
    s.policy_temp = num(d, "policy_temp", s.policy_temp);
    s.q_n0 = num(d, "q_n0", s.q_n0);
    s.c1 = num(d, "c1", s.c1);
    s.c2 = num(d, "c2", s.c2);
    s.explore_growth = num(d, "explore_growth", s.explore_growth);
    s.explore_base = num(d, "explore_base", s.explore_base);
    s.forcing = (int)num(d, "forcing", s.forcing);
    s.forcing_after = num(d, "forcing_after", s.forcing_after);
    s.forcing_apart = num(d, "forcing_apart", s.forcing_apart) != 0.0;
    s.collide = (int)num(d, "collide", s.collide);
    s.cache_progress = num(d, "cache_progress", s.cache_progress) != 0.0;
    s.fill_once = num(d, "fill_once", s.fill_once) != 0.0;
    s.draw_plies = (int)num(d, "draw_plies", s.draw_plies);
    s.lcb = num(d, "lcb", s.lcb) != 0.0;
    s.lcb_alpha = num(d, "lcb_alpha", s.lcb_alpha);
    s.lcb_floor = num(d, "lcb_floor", s.lcb_floor);
    s.prior_fallback = num(d, "prior_fallback", s.prior_fallback) != 0.0;
    s.cache_size = (size_t)num(d, "cache_size", (double)s.cache_size);
    s.use_tb = num(d, "use_tb", 0.0) != 0.0 && tbfn != Py_None;
    s.tb_pieces = (int)num(d, "tb_pieces", s.tb_pieces);
    s.pipeline = (int)num(d, "pipeline", 0.0);
    s.prof = num(d, "_prof", 0.0) != 0.0;
    if (s.pipeline < 0 || s.pipeline > 2) {
        delete inner;
        PyErr_SetString(PyExc_ValueError, "pipeline must be 0, 1 or 2");
        return -1;
    }
    delete self->tree;
    delete self->net;
    self->net = new PyNet(fn == Py_None ? nullptr : fn, tbfn == Py_None ? nullptr : tbfn, inner);
    self->tree = new Tree(self->net, s);
    if (s.prof) {
        try {
            self->net->timers_on(true);
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return -1;
        }
    }
    return 0;
}

void Tree_dealloc(TreeObject* self)
{
    delete self->tree;
    delete self->net;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* Tree_reset(TreeObject* self, PyObject*)
{
    self->tree->reset();
    Py_RETURN_NONE;
}

PyObject* Tree_begin(TreeObject* self, PyObject* args)
{
    const char* fen;
    PyObject* moves;
    int reuse;
    if (!PyArg_ParseTuple(args, "sOp", &fen, &moves, &reuse)) return nullptr;
    Pos p;
    if (!parse_fen(fen, p)) { PyErr_SetString(PyExc_ValueError, "bad FEN"); return nullptr; }
    PyObject* seq = PySequence_Fast(moves, "moves must be a sequence of UCI strings");
    if (!seq) return nullptr;
    std::vector<Pos> history;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; i++) {
        const char* u = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(seq, i));
        if (!u) { Py_DECREF(seq); return nullptr; }
        const int c = parse_uci(p, u);
        if (c < 0) { Py_DECREF(seq); PyErr_Format(PyExc_ValueError, "illegal move %s", u); return nullptr; }
        history.push_back(p);
        push(p, c);
    }
    Py_DECREF(seq);
    double carried;
    try {
        carried = self->tree->begin(p, history, reuse != 0);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    return PyFloat_FromDouble(carried);
}

PyObject* Tree_discount(TreeObject* self, PyObject* args)
{
    double d;
    int q0;
    if (!PyArg_ParseTuple(args, "dp", &d, &q0)) return nullptr;
    self->tree->discount(d, q0 != 0);
    Py_RETURN_NONE;
}

PyObject* Tree_needs_eval(TreeObject* self, PyObject*)
{
    return PyBool_FromLong(self->tree->needs_eval());
}

PyObject* Tree_expand_root(TreeObject* self, PyObject*)
{
    try {
        self->tree->expand_root();
    } catch (const PyError&) {
        restore(self);
        return nullptr;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Tree_step(TreeObject* self, PyObject* args)
{
    double target, deadline;
    if (!PyArg_ParseTuple(args, "dd", &target, &deadline)) return nullptr;
    if (!self->tree->root()) { PyErr_SetString(PyExc_RuntimeError, "step before begin"); return nullptr; }
    int code = 0;
    int failed = 0;   // 1 Python error, 2 C++ error
    std::string what;
    Py_BEGIN_ALLOW_THREADS
    try {
        code = self->tree->step(target, deadline);
    } catch (const PyError&) {
        failed = 1;
    } catch (const std::exception& e) {
        failed = 2;
        what = e.what();
    }
    Py_END_ALLOW_THREADS
    if (failed == 1) { restore(self); return nullptr; }
    if (failed == 2) { PyErr_SetString(PyExc_RuntimeError, what.c_str()); return nullptr; }
    return PyLong_FromLong(code);
}

PyObject* Tree_finish(TreeObject* self, PyObject*)
{
    int failed = 0;   // 1 Python error, 2 C++ error
    std::string what;
    Py_BEGIN_ALLOW_THREADS
    try {
        self->tree->finish();
    } catch (const PyError&) {
        failed = 1;
    } catch (const std::exception& e) {
        failed = 2;
        what = e.what();
    }
    Py_END_ALLOW_THREADS
    if (failed == 1) { restore(self); return nullptr; }
    if (failed == 2) { PyErr_SetString(PyExc_RuntimeError, what.c_str()); return nullptr; }
    Py_RETURN_NONE;
}

PyObject* Tree_held(TreeObject* self, PyObject*)
{
    return PyFloat_FromDouble(self->tree->held());
}

PyObject* Tree_in_flight(TreeObject* self, PyObject*)
{
    return PyBool_FromLong(self->tree->in_flight());
}

PyObject* Tree_prof(TreeObject* self, PyObject*)
{
    const Prof& p = self->tree->prof();
    PyObject* out = Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:L}", "descend", p.descend, "leaf", p.leaf,
                                  "encode", p.encode, "net", p.net, "submit", p.submit, "collect", p.collect,
                                  "expand", p.expand, "step", p.step, "steps", p.steps);
    if (!out) return nullptr;
    std::vector<std::pair<std::string, double>> t;
    try {
        t = self->net->timers();
    } catch (const std::exception& e) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    for (auto& kv : t) {
        PyObject* v = PyFloat_FromDouble(kv.second);
        const std::string k = "net_" + kv.first;
        if (!v || PyDict_SetItemString(out, k.c_str(), v) < 0) {
            Py_XDECREF(v);
            Py_DECREF(out);
            return nullptr;
        }
        Py_DECREF(v);
    }
    return out;
}

PyObject* Tree_choose(TreeObject* self, PyObject*)
{
    return PyLong_FromLong(self->tree->choose());
}

PyObject* Tree_root(TreeObject* self, PyObject*)
{
    Node* r = self->tree->root();
    if (!r) Py_RETURN_NONE;
    PyObject* moves = PyList_New((Py_ssize_t)r->moves.size());
    for (size_t i = 0; i < r->moves.size(); i++)
        PyList_SET_ITEM(moves, (Py_ssize_t)i, PyUnicode_FromString(uci(r->moves[i]).c_str()));
    PyObject* term = r->has_term ? PyFloat_FromDouble(r->term) : (Py_INCREF(Py_None), Py_None);
    PyObject* kids = PyList_New((Py_ssize_t)r->kids.size());
    for (size_t i = 0; i < r->kids.size(); i++)
        PyList_SET_ITEM(kids, (Py_ssize_t)i, r->kids[i] ? PyFloat_FromDouble(r->kids[i]->sum_n) : (Py_INCREF(Py_None), Py_None));
    PyObject* out = Py_BuildValue(
        "{s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:d,s:d,s:O,s:N,s:N,s:d}",
        "moves", moves, "N", bytes_of(r->N), "W", bytes_of(r->W), "V2", bytes_of(r->V2), "F", bytes_of(r->F),
        "X", bytes_of(r->X), "Q0", bytes_of(r->Q0), "P", bytes_of(r->P), "sum_n", r->sum_n, "q0w", r->q0w,
        "expanded", r->is_expanded ? Py_True : Py_False, "term", term, "kids_n", kids, "value", r->value);
    return out;
}

PyObject* Tree_pv(TreeObject* self, PyObject* args)
{
    int first, depth;
    if (!PyArg_ParseTuple(args, "ii", &first, &depth)) return nullptr;
    std::vector<int> line = self->tree->pv(first, depth);
    PyObject* out = PyList_New((Py_ssize_t)line.size());
    for (size_t i = 0; i < line.size(); i++)
        PyList_SET_ITEM(out, (Py_ssize_t)i, PyUnicode_FromString(uci(line[i]).c_str()));
    return out;
}

PyObject* Tree_tree_nodes(TreeObject* self, PyObject*)
{
    return PyLong_FromLongLong(self->tree->tree_nodes());
}

PyObject* Tree_stats(TreeObject* self, PyObject*)
{
    const Stats& s = self->tree->stats();
    return Py_BuildValue("{s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:n,s:n}",
                         "evals", s.evals, "batches", s.batches, "cache_hits", s.cache_hits,
                         "collisions", s.collisions, "forced", s.forced, "forced_apart", s.forced_apart,
                         "repetitions", s.repetitions, "lcb_overrode", s.lcb_overrode,
                         "discounted", s.discounted, "reused", s.reused, "tablebase", s.tablebase,
                         "pool", (Py_ssize_t)self->tree->pool_size(), "cache", (Py_ssize_t)self->tree->cache_len());
}

PyMethodDef Tree_methods[] = {
    {"reset", (PyCFunction)Tree_reset, METH_NOARGS, "a new game"},
    {"begin", (PyCFunction)Tree_begin, METH_VARARGS, "begin(fen, moves, reuse) -> carried visits"},
    {"discount", (PyCFunction)Tree_discount, METH_VARARGS, "discount(d, q0)"},
    {"needs_eval", (PyCFunction)Tree_needs_eval, METH_NOARGS, "the root needs its forward pass"},
    {"expand_root", (PyCFunction)Tree_expand_root, METH_NOARGS, "the root's forward pass"},
    {"step", (PyCFunction)Tree_step, METH_VARARGS, "step(target, deadline) -> 0 batch, 1 nothing, 2 saturated"},
    {"choose", (PyCFunction)Tree_choose, METH_NOARGS, "the root move's index (MCTS._choose)"},
    {"root", (PyCFunction)Tree_root, METH_NOARGS, "the root's statistics"},
    {"pv", (PyCFunction)Tree_pv, METH_VARARGS, "pv(first, depth) -> [uci]"},
    {"tree_nodes", (PyCFunction)Tree_tree_nodes, METH_NOARGS, "MCTS.tree_nodes"},
    {"stats", (PyCFunction)Tree_stats, METH_NOARGS, "the counters since reset"},
    {"finish", (PyCFunction)Tree_finish, METH_NOARGS, "back up the pipeline's batch still on the card"},
    {"held", (PyCFunction)Tree_held, METH_NOARGS, "the held collisions of the batch on the card"},
    {"in_flight", (PyCFunction)Tree_in_flight, METH_NOARGS, "a pipeline batch is on the card"},
    {"prof", (PyCFunction)Tree_prof, METH_NOARGS, "seconds by part of a step (settings _prof)"},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject TreeType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "mayfly_cppsearch", "MayFly's C++ search tree (milestone 2)", -1,
                      nullptr};

}  // namespace

PyMODINIT_FUNC PyInit_mayfly_cppsearch(void)
{
    TreeType.tp_name = "mayfly_cppsearch.Tree";
    TreeType.tp_basicsize = sizeof(TreeObject);
    TreeType.tp_flags = Py_TPFLAGS_DEFAULT;
    TreeType.tp_new = PyType_GenericNew;
    TreeType.tp_init = (initproc)Tree_init;
    TreeType.tp_dealloc = (destructor)Tree_dealloc;
    TreeType.tp_methods = Tree_methods;
    if (PyType_Ready(&TreeType) < 0) return nullptr;
    PyObject* m = PyModule_Create(&moddef);
    if (!m) return nullptr;
    Py_INCREF(&TreeType);
    if (PyModule_AddObject(m, "Tree", (PyObject*)&TreeType) < 0) {
        Py_DECREF(&TreeType);
        Py_DECREF(m);
        return nullptr;
    }
    return m;
}
