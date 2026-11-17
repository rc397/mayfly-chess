/* MayFly's descent, unwind and backup in C (timing_fixes, v5). It walks the existing Python Node objects
   and their numpy float32 rows. A prototype of this descent ran 8 to 9 times faster than _descend with the
   compiled pick, bit for bit.

   descend(root, params, stats, collided, terminal_cb, fm_cb, pick_cb) -> (path, parent, a) or None
       params = (q_n0, c1, c2, explore_growth, explore_base, forcing, forcing_after, forcing_apart)
       MCTS._descend for the selection v5 plays: _pick_fast's case (q_init, puct_v, no q_floor, no
       sigma_done, no mlh, no root check, no shortlist) with a virtual mean and no graph search. The pick
       is pick.c's arithmetic, operation for operation, and the forcing guard is _pick_fast's. The parts
       that stay in Python are called back. terminal_cb(path, node) backs up a terminal as _descend does
       (mate_dist, moves-left), fm_cb(node) builds node.FM the first time the forcing guard asks, and
       pick_cb(node, is_root, ply1) -> (a, q, forced) picks at a node the fast arithmetic cannot serve (no
       q head row, no prior weight, a row that is not contiguous float32), which is _pick_fast itself. A
       collision is appended to `collided` (held, collide 2) or, with None there, taken back as _unwind.
   unwind(path)                  MCTS._unwind
   backup(path, leaf_value)      MCTS._backup for power_p 1 with proofs and moves-left off; leaf_value
                                 is a Python float, as mcts.py checks before it calls, so v is a double
                                 exactly as there
   expand_rows(logits, idx, temp, k)   _expand's float64 softmax prior and its five zero rows, behind a
                                 check against numpy when mcts.py loads it
   Every float32 operation is numpy's, in the same order, one rounding each (NEP 50, a Python float
   meeting a float32 array is cast to float32 first). W[a] += x is W[a] + (float)x, and _backup's
   W[a] += v - x is W[a] + (float)(v - x) with v - x a double. sqrt is correctly rounded on both sides,
   and log is the same C runtime function Python's math.log calls. MSVC /fp:precise with no /arch flag
   never fuses a multiply and an add (setup.py). harness.py proves it on the stub and the real network. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* slot offsets of mcts.Node, read once from its member descriptors */
static Py_ssize_t o_moves, o_P, o_N, o_W, o_V2, o_kids, o_term, o_sum_n, o_pending, o_Q0, o_F, o_X, o_q0w, o_FM;
static PyTypeObject *NodeType = NULL;
static PyObject *K_collisions, *K_forced, *K_forced_apart, *ONE;

static Py_ssize_t slot_off(PyObject *cls, const char *name)
{
    PyObject *d = PyObject_GetAttrString(cls, name);
    if (!d) return -1;
    if (!PyObject_TypeCheck(d, &PyMemberDescr_Type)) {
        Py_DECREF(d);
        PyErr_Format(PyExc_TypeError, "Node.%s is not a slot", name);
        return -1;
    }
    Py_ssize_t off = ((PyMemberDescrObject *)d)->d_member->offset;
    Py_DECREF(d);
    return off;
}

static PyObject *setup(PyObject *self, PyObject *cls)
{
    if (!PyType_Check(cls)) { PyErr_SetString(PyExc_TypeError, "setup(Node)"); return NULL; }
#define S(v, n) if ((v = slot_off(cls, n)) < 0) return NULL;
    S(o_moves, "moves") S(o_P, "P") S(o_N, "N") S(o_W, "W") S(o_V2, "V2") S(o_kids, "kids") S(o_term, "term")
    S(o_sum_n, "sum_n") S(o_pending, "pending") S(o_Q0, "Q0") S(o_F, "F") S(o_X, "X") S(o_q0w, "q0w") S(o_FM, "FM")
#undef S
    K_collisions = PyUnicode_InternFromString("collisions");
    K_forced = PyUnicode_InternFromString("forced");
    K_forced_apart = PyUnicode_InternFromString("forced_apart");
    ONE = PyLong_FromLong(1);
    if (!K_collisions || !K_forced || !K_forced_apart || !ONE) return NULL;
    Py_INCREF(cls);
    Py_XDECREF((PyObject *)NodeType);
    NodeType = (PyTypeObject *)cls;
    Py_RETURN_NONE;
}

#define SLOT(o, off) (*(PyObject **)((char *)(o) + (off)))

/* the data of a 1-D C-contiguous float32 array of length k (any length when k < 0), else NULL */
static float *f32(PyObject *a, Py_ssize_t k)
{
    if (!a || !PyArray_Check(a)) return NULL;
    PyArrayObject *x = (PyArrayObject *)a;
    if (PyArray_TYPE(x) != NPY_FLOAT32 || PyArray_NDIM(x) != 1 || !PyArray_IS_C_CONTIGUOUS(x)) return NULL;
    if (k >= 0 && PyArray_DIM(x, 0) != k) return NULL;
    return (float *)PyArray_DATA(x);
}

static Py_ssize_t len1(PyObject *a) { return PyArray_DIM((PyArrayObject *)a, 0); }

/* stats[key] = stats.get(key, 0) + 1 */
static int bump(PyObject *stats, PyObject *key)
{
    PyObject *v = PyDict_GetItemWithError(stats, key);
    if (!v && PyErr_Occurred()) return -1;
    PyObject *nv;
    if (v) {
        nv = PyNumber_Add(v, ONE);
    } else {
        Py_INCREF(ONE);
        nv = ONE;
    }
    if (!nv) return -1;
    int r = PyDict_SetItem(stats, key, nv);
    Py_DECREF(nv);
    return r;
}

/* node.sum_n = node.sum_n + d, on whatever number it holds (an int 0 at birth, an int after the ponder
   discount, a float after a visit), exactly as `node.sum_n += 1.0` and `-= 1.0` do */
static int add_sum(PyObject *node, double d)
{
    PyObject *old = SLOT(node, o_sum_n);
    if (!old) { PyErr_SetString(PyExc_AttributeError, "sum_n"); return -1; }
    PyObject *dd = PyFloat_FromDouble(d);
    if (!dd) return -1;
    PyObject *nv = PyNumber_Add(old, dd);
    Py_DECREF(dd);
    if (!nv) return -1;
    SLOT(node, o_sum_n) = nv;
    Py_DECREF(old);
    return 0;
}

/* one step of a path: (node, a, x), x None for a forced visit kept apart from the root */
static int step_parts(PyObject *st, PyObject **nd, Py_ssize_t *a, PyObject **px)
{
    if (!PyTuple_Check(st) || PyTuple_GET_SIZE(st) != 3) { PyErr_SetString(PyExc_TypeError, "a path step"); return -1; }
    *nd = PyTuple_GET_ITEM(st, 0);
    *px = PyTuple_GET_ITEM(st, 2);
    if (Py_TYPE(*nd) != NodeType) { PyErr_SetString(PyExc_TypeError, "a path step's node"); return -1; }
    *a = PyLong_AsSsize_t(PyTuple_GET_ITEM(st, 1));
    if (*a == -1 && PyErr_Occurred()) return -1;
    return 0;
}

/* MCTS._unwind, N[a] -= 1.0, W[a] -= x, sum_n -= 1.0, F[a] -= 1.0, X[a] -= x, with x cast to float32 */
static int do_unwind(PyObject *path)
{
    if (!PyList_Check(path)) { PyErr_SetString(PyExc_TypeError, "path"); return -1; }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(path); i++) {
        PyObject *nd, *px;
        Py_ssize_t a;
        if (step_parts(PyList_GET_ITEM(path, i), &nd, &a, &px)) return -1;
        if (px == Py_None) { if (add_sum(nd, -1.0)) return -1; continue; }
        float *n = f32(SLOT(nd, o_N), -1);
        Py_ssize_t k = n ? len1(SLOT(nd, o_N)) : 0;
        float *w = f32(SLOT(nd, o_W), k), *f = f32(SLOT(nd, o_F), k), *X = f32(SLOT(nd, o_X), k);
        if (!n || !w || !f || !X || a < 0 || a >= k) { PyErr_SetString(PyExc_TypeError, "unwind: node rows"); return -1; }
        double xd = PyFloat_AsDouble(px);
        if (xd == -1.0 && PyErr_Occurred()) return -1;
        const float x = (float)xd;
        n[a] = n[a] - 1.0f;
        w[a] = w[a] - x;
        if (add_sum(nd, -1.0)) return -1;
        f[a] = f[a] - 1.0f;
        X[a] = X[a] - x;
    }
    return 0;
}

static PyObject *unwind(PyObject *self, PyObject *path)
{
    if (!NodeType) { PyErr_SetString(PyExc_RuntimeError, "mayfly_walk.setup(Node) first"); return NULL; }
    if (do_unwind(path)) return NULL;
    Py_RETURN_NONE;
}

/* MCTS._backup for power_p 1, proofs off and moves-left off:
       W[a] += v - x    ->  W[a] = W[a] + (float)(v - x)
       V2[a] += v * v   ->  V2[a] = V2[a] + (float)(v * v)
       F[a] -= 1.0      ->  F[a] = F[a] - 1.0f
       X[a] -= x        ->  X[a] = X[a] - (float)x
   v and x doubles, as the Python floats they are there. The sign turns at every step, kept-apart ones too. */
static PyObject *backup(PyObject *self, PyObject *args)
{
    PyObject *path;
    double leaf;
    if (!PyArg_ParseTuple(args, "Od", &path, &leaf)) return NULL;
    if (!NodeType) { PyErr_SetString(PyExc_RuntimeError, "mayfly_walk.setup(Node) first"); return NULL; }
    if (!PyList_Check(path)) { PyErr_SetString(PyExc_TypeError, "path"); return NULL; }
    double v = -leaf;
    for (Py_ssize_t i = PyList_GET_SIZE(path) - 1; i >= 0; i--) {
        PyObject *nd, *px;
        Py_ssize_t a;
        if (step_parts(PyList_GET_ITEM(path, i), &nd, &a, &px)) return NULL;
        if (px == Py_None) { v = -v; continue; }
        float *w = f32(SLOT(nd, o_W), -1);
        Py_ssize_t k = w ? len1(SLOT(nd, o_W)) : 0;
        float *v2 = f32(SLOT(nd, o_V2), k), *f = f32(SLOT(nd, o_F), k), *X = f32(SLOT(nd, o_X), k);
        if (!w || !v2 || !f || !X || a < 0 || a >= k) { PyErr_SetString(PyExc_TypeError, "backup: node rows"); return NULL; }
        double x = PyFloat_AsDouble(px);
        if (x == -1.0 && PyErr_Occurred()) return NULL;
        w[a] = w[a] + (float)(v - x);
        v2[a] = v2[a] + (float)(v * v);
        f[a] = f[a] - 1.0f;
        X[a] = X[a] - (float)x;
        v = -v;
    }
    Py_RETURN_NONE;
}

static PyObject *descend(PyObject *self, PyObject *args)
{
    PyObject *root, *stats, *collided, *terminal_cb, *fm_cb, *pick_cb;
    double q_n0, c1, c2, eg, eb, forcing_after;
    int forcing, apart;
    if (!PyArg_ParseTuple(args, "O(dddddidp)OOOOO", &root, &q_n0, &c1, &c2, &eg, &eb, &forcing, &forcing_after,
                          &apart, &stats, &collided, &terminal_cb, &fm_cb, &pick_cb))
        return NULL;
    if (!NodeType) { PyErr_SetString(PyExc_RuntimeError, "mayfly_walk.setup(Node) first"); return NULL; }
    if (!PyDict_Check(stats)) { PyErr_SetString(PyExc_TypeError, "stats"); return NULL; }
    if (collided != Py_None && !PyList_Check(collided)) { PyErr_SetString(PyExc_TypeError, "collided"); return NULL; }
    PyObject *path = PyList_New(0);
    if (!path) return NULL;
    const float c1f = (float)c1, c2f = (float)(2.0 * c2);
    PyObject *node = root;
    Py_INCREF(node);
    PyObject *result = NULL;
    for (;;) {
        if (Py_TYPE(node) != NodeType) { PyErr_SetString(PyExc_TypeError, "descend: not a Node"); goto fail; }
        PyObject *term = SLOT(node, o_term);
        if (term && term != Py_None) {
            PyObject *r = PyObject_CallFunctionObjArgs(terminal_cb, path, node, NULL);
            if (!r) goto fail;
            Py_DECREF(r);
            Py_INCREF(Py_None);
            result = Py_None;
            goto done;
        }
        PyObject *pend = SLOT(node, o_pending);
        int pending = pend ? PyObject_IsTrue(pend) : 0;
        if (pending < 0) goto fail;
        if (pending) {
            if (bump(stats, K_collisions)) goto fail;
            if (collided != Py_None) {
                PyObject *t = PyTuple_Pack(2, path, node);
                if (!t) goto fail;
                int r = PyList_Append(collided, t);
                Py_DECREF(t);
                if (r) goto fail;
            } else if (do_unwind(path)) {
                goto fail;
            }
            Py_INCREF(Py_None);
            result = Py_None;
            goto done;
        }
        const Py_ssize_t plen = PyList_GET_SIZE(path);
        PyObject *oN = SLOT(node, o_N);
        float *n = f32(oN, -1);
        const Py_ssize_t k = n ? len1(oN) : 0;
        float *w = f32(SLOT(node, o_W), k), *p = f32(SLOT(node, o_P), k), *q0 = f32(SLOT(node, o_Q0), k);
        float *v2 = f32(SLOT(node, o_V2), k), *F = f32(SLOT(node, o_F), k), *X = f32(SLOT(node, o_X), k);
        PyObject *oq0w = SLOT(node, o_q0w), *osum = SLOT(node, o_sum_n);
        if (!oq0w || !osum) { PyErr_SetString(PyExc_AttributeError, "q0w or sum_n"); goto fail; }
        const double q0w = PyFloat_AsDouble(oq0w);
        if (q0w == -1.0 && PyErr_Occurred()) goto fail;
        const double n0 = q_n0 * q0w;
        const double sum_n = PyFloat_AsDouble(osum);
        if (sum_n == -1.0 && PyErr_Occurred()) goto fail;
        Py_ssize_t a = -1;
        double xd = 0.0;
        int forced = 0;
        if (!n || k <= 0 || !w || !p || !q0 || !v2 || !F || !X || n0 <= 0.0) {
            /* _pick_fast hands such a node to _pick; this hands it to _pick_fast */
            PyObject *r = PyObject_CallFunction(pick_cb, "OOO", node, node == root ? Py_True : Py_False,
                                                plen == 1 ? Py_True : Py_False);
            if (!r) goto fail;
            PyObject *ra, *rq, *rf;
            if (!PyArg_ParseTuple(r, "OOO", &ra, &rq, &rf)) { Py_DECREF(r); goto fail; }
            a = PyLong_AsSsize_t(ra);
            xd = PyFloat_AsDouble(rq);
            forced = PyObject_IsTrue(rf);
            Py_DECREF(r);
            if (PyErr_Occurred() || forced < 0) goto fail;
            oN = SLOT(node, o_N);
            n = f32(oN, -1);
            Py_ssize_t k2 = n ? len1(oN) : 0;
            w = f32(SLOT(node, o_W), k2);
            F = f32(SLOT(node, o_F), k2);
            X = f32(SLOT(node, o_X), k2);
            if (!n || !w || !F || !X || a < 0 || a >= k2) { PyErr_SetString(PyExc_TypeError, "descend: node rows"); goto fail; }
        } else {
            const float n0f = (float)n0;
            int moves_left = 0;
            if (plen == 1 && forcing > 0 && sum_n >= forcing_after) {
                PyObject *mv = SLOT(node, o_moves);
                moves_left = mv ? PyObject_IsTrue(mv) : 0;
                if (moves_left < 0) goto fail;
            }
            if (moves_left) {
                /* _pick_fast's forcing guard, the least-visited capture or check below `forcing` visits */
                PyObject *fm = SLOT(node, o_FM);
                if (!fm || fm == Py_None) {
                    PyObject *r = PyObject_CallFunctionObjArgs(fm_cb, node, NULL);
                    if (!r) goto fail;
                    Py_DECREF(r);
                    fm = SLOT(node, o_FM);
                }
                if (!fm || !PyArray_Check(fm) || PyArray_TYPE((PyArrayObject *)fm) != NPY_BOOL
                        || PyArray_NDIM((PyArrayObject *)fm) != 1 || PyArray_DIM((PyArrayObject *)fm, 0) != k
                        || !PyArray_IS_C_CONTIGUOUS((PyArrayObject *)fm)) {
                    PyErr_SetString(PyExc_TypeError, "descend: node.FM is not a bool row of the node's moves");
                    goto fail;
                }
                const npy_bool *m = (const npy_bool *)PyArray_DATA((PyArrayObject *)fm);
                const float lim = (float)forcing;
                float least = 0.0f;
                for (Py_ssize_t i = 0; i < k; i++)
                    if (m[i] && n[i] < lim && (a < 0 || n[i] < least)) { a = i; least = n[i]; }
                if (a >= 0) {
                    float t = n0f * q0[a];
                    t = t + w[a];
                    float den = n0f + n[a];
                    xd = (double)(t / den);
                    forced = 1;
                    if (bump(stats, K_forced)) goto fail;
                }
            }
            if (a < 0) {
                /* pick.c, operation for operation, with the scalars _pick_fast hands it */
                const double N = sum_n >= 1.0 ? sum_n : 1.0;
                const double grow = eg > 0.0 ? 1.0 + eg * log((N + eb) / eb) : 0.0;
                const float sqf = (float)sqrt(N), lgf = (float)log(N + 1.0), grf = (float)grow;
                const int growing = grow > 0.0;
                Py_ssize_t best = -1, first_nan = -1;
                float bestu = 0.0f, bestq = 0.0f, nanq = 0.0f;
                for (Py_ssize_t i = 0; i < k; i++) {
                    float t = n0f * q0[i];
                    t = t + w[i];
                    float den = n0f + n[i];
                    float q = t / den;
                    float sd;
                    if (n[i] > 0.0f) {
                        float mean = w[i] / n[i];
                        float vv = v2[i] / n[i];
                        float mm = mean * mean;
                        vv = vv - mm;
                        vv = (vv < 0.0f) ? 0.0f : vv;
                        sd = sqrtf(vv);
                    } else {
                        sd = 1.0f;
                    }
                    float d = 1.0f + n[i];
                    float t1 = c1f * p[i];
                    t1 = t1 * sd;
                    t1 = t1 * sqf;
                    t1 = t1 / d;
                    float t2 = c2f * p[i];
                    t2 = t2 * lgf;
                    t2 = t2 / d;
                    float u = t1 + t2;
                    if (growing)
                        u = u * grf;
                    u = u + q;
                    if (u != u) {
                        if (first_nan < 0) { first_nan = i; nanq = q; }
                        continue;
                    }
                    if (best < 0 || u > bestu) { best = i; bestu = u; bestq = q; }
                }
                if (first_nan >= 0) { a = first_nan; xd = (double)nanq; }
                else { a = best; xd = (double)bestq; }
            }
        }
        /* the virtual mean, where x is the q selection read, a Python float, which numpy adds as a float32 */
        const float x = (float)xd;
        n[a] = n[a] + 1.0f;
        w[a] = w[a] + x;
        if (add_sum(node, 1.0)) goto fail;
        F[a] = F[a] + 1.0f;
        X[a] = X[a] + x;
        PyObject *pa = PyLong_FromSsize_t(a);
        if (!pa) goto fail;
        PyObject *px = PyFloat_FromDouble(xd);
        if (!px) { Py_DECREF(pa); goto fail; }
        PyObject *st = PyTuple_Pack(3, node, pa, px);
        Py_DECREF(px);
        if (!st || PyList_Append(path, st)) { Py_XDECREF(st); Py_DECREF(pa); goto fail; }
        Py_DECREF(st);
        if (apart && forced && plen + 1 == 2) {
            /* kept apart from the root, so the root step's visit comes back out of its edge as _unwind would
               take it, leaving only root.sum_n (the budget), and x None marks the step */
            PyObject *s0 = PyList_GET_ITEM(path, 0), *r, *rx;
            Py_ssize_t ra;
            if (step_parts(s0, &r, &ra, &rx) || rx == Py_None) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "descend: root step");
                Py_DECREF(pa);
                goto fail;
            }
            float *rn = f32(SLOT(r, o_N), -1);
            Py_ssize_t rk = rn ? len1(SLOT(r, o_N)) : 0;
            float *rw = f32(SLOT(r, o_W), rk), *rf = f32(SLOT(r, o_F), rk), *rX = f32(SLOT(r, o_X), rk);
            double rxd = PyFloat_AsDouble(rx);
            if (!rn || !rw || !rf || !rX || ra < 0 || ra >= rk || (rxd == -1.0 && PyErr_Occurred())) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "descend: root rows");
                Py_DECREF(pa);
                goto fail;
            }
            const float rxf = (float)rxd;
            rn[ra] = rn[ra] - 1.0f;
            rw[ra] = rw[ra] - rxf;
            rf[ra] = rf[ra] - 1.0f;
            rX[ra] = rX[ra] - rxf;
            PyObject *ns = PyTuple_Pack(3, r, PyTuple_GET_ITEM(s0, 1), Py_None);
            if (!ns) { Py_DECREF(pa); goto fail; }
            if (PyList_SetItem(path, 0, ns)) { Py_DECREF(pa); goto fail; }     /* steals ns */
            if (bump(stats, K_forced_apart)) { Py_DECREF(pa); goto fail; }
        }
        PyObject *kids = SLOT(node, o_kids);
        if (!kids || !PyList_Check(kids) || a >= PyList_GET_SIZE(kids)) {
            PyErr_SetString(PyExc_TypeError, "descend: node.kids");
            Py_DECREF(pa);
            goto fail;
        }
        PyObject *kid = PyList_GET_ITEM(kids, a);
        if (kid == Py_None) {
            result = PyTuple_Pack(3, path, node, pa);
            Py_DECREF(pa);
            goto done;
        }
        Py_DECREF(pa);
        Py_INCREF(kid);
        Py_DECREF(node);
        node = kid;
    }
fail:
    result = NULL;
done:
    Py_DECREF(node);
    Py_DECREF(path);
    return result;
}

/* numpy's DOUBLE_pairwise_sum (numpy/_core/src/umath/loops_utils.h.src) over a contiguous row, which is what a
   float64 add.reduce of a 1-D array computes, added to the reduction's identity 0.0 */
static double pairwise(const double *a, Py_ssize_t n)
{
    if (n < 8) {
        double res = -0.0;
        for (Py_ssize_t i = 0; i < n; i++)
            res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        Py_ssize_t i;
        for (int j = 0; j < 8; j++)
            r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++)
                r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++)
            res += a[i];
        return res;
    }
    Py_ssize_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

/* expand_rows(logits, idx, temp, k) -> (P, N, W, V2, F, X), or None for numpy's own path
   _expand's prior and its five zero rows of length k (timing_fixes), the numbers numpy computes:
       z = (logits if idx is None else logits[idx]).astype(np.float64)
       if temp != 1.0: z /= temp
       z -= z.max(); p = np.exp(z); P = (p / p.sum()).astype(np.float32)
   None when np.add.reduce(z) is not finite (a NaN or an infinity, which _expand's guard handles) and for
   anything but a 1-D float32 row. exp is the C runtime's, which is numpy's float64 exp on a machine
   without AVX-512, and the sums are numpy's pairwise sum. mcts.py compares both, and whole rows, with
   numpy when it loads this module, and leaves this function unused on a machine where any differs. */
static PyObject *expand_rows(PyObject *self, PyObject *args)
{
    PyObject *ol, *oi;
    double temp;
    Py_ssize_t krows;
    if (!PyArg_ParseTuple(args, "OOdn", &ol, &oi, &temp, &krows)) return NULL;
    if (!PyArray_Check(ol) || PyArray_TYPE((PyArrayObject *)ol) != NPY_FLOAT32 || PyArray_NDIM((PyArrayObject *)ol) != 1)
        Py_RETURN_NONE;
    PyArrayObject *L = (PyArrayObject *)ol;
    const Py_ssize_t nl = PyArray_DIM(L, 0), sl = PyArray_STRIDE(L, 0);
    const char *lb = PyArray_BYTES(L);
    PyArrayObject *IX = NULL;
    Py_ssize_t k = nl;
    if (oi != Py_None) {
        IX = (PyArrayObject *)PyArray_FROMANY(oi, NPY_INT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (!IX) return NULL;
        k = PyArray_DIM(IX, 0);
    }
    if (k <= 0 || krows < 0) { Py_XDECREF(IX); Py_RETURN_NONE; }
    double zb[320];
    double *z = k <= 320 ? zb : (double *)PyMem_Malloc((size_t)k * sizeof(double));
    if (!z) { Py_XDECREF(IX); return PyErr_NoMemory(); }
    const npy_int64 *ix = IX ? (const npy_int64 *)PyArray_DATA(IX) : NULL;
    PyObject *res = NULL;
    for (Py_ssize_t i = 0; i < k; i++) {
        Py_ssize_t j = ix ? (Py_ssize_t)ix[i] : i;
        if (j < 0) j += nl;                                /* numpy's negative index */
        if (j < 0 || j >= nl) { PyErr_SetString(PyExc_IndexError, "expand_rows: index out of bounds"); goto out; }
        z[i] = (double)*(const float *)(lb + j * sl);
    }
    if (!isfinite(0.0 + pairwise(z, k))) { Py_INCREF(Py_None); res = Py_None; goto out; }
    if (temp != 1.0)
        for (Py_ssize_t i = 0; i < k; i++)
            z[i] /= temp;
    double mx = z[0];
    for (Py_ssize_t i = 1; i < k; i++)
        if (z[i] > mx) mx = z[i];
    for (Py_ssize_t i = 0; i < k; i++)
        z[i] = exp(z[i] - mx);
    const double s = 0.0 + pairwise(z, k);
    npy_intp dk[1] = {(npy_intp)k}, dr[1] = {(npy_intp)krows};
    PyObject *P = PyArray_SimpleNew(1, dk, NPY_FLOAT32);
    if (!P) goto out;
    float *pp = (float *)PyArray_DATA((PyArrayObject *)P);
    for (Py_ssize_t i = 0; i < k; i++)
        pp[i] = (float)(z[i] / s);
    PyObject *r[5] = {NULL, NULL, NULL, NULL, NULL};
    for (int q = 0; q < 5; q++) {
        r[q] = PyArray_ZEROS(1, dr, NPY_FLOAT32, 0);
        if (!r[q]) {
            Py_DECREF(P);
            for (int u = 0; u < q; u++) Py_DECREF(r[u]);
            goto out;
        }
    }
    res = PyTuple_Pack(6, P, r[0], r[1], r[2], r[3], r[4]);
    Py_DECREF(P);
    for (int q = 0; q < 5; q++) Py_DECREF(r[q]);
out:
    if (z != zb) PyMem_Free(z);
    Py_XDECREF(IX);
    return res;
}

/* the two primitives on their own, for mcts.py's check at load, where a float32 rounding cannot hide a
   difference (exp over a float64 row, and the pairwise sum of one) */
static PyObject *check_exp(PyObject *self, PyObject *o)
{
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(o, NPY_FLOAT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!a) return NULL;
    npy_intp d[1] = {PyArray_DIM(a, 0)};
    PyObject *out = PyArray_SimpleNew(1, d, NPY_FLOAT64);
    if (out) {
        const double *x = (const double *)PyArray_DATA(a);
        double *y = (double *)PyArray_DATA((PyArrayObject *)out);
        for (npy_intp i = 0; i < d[0]; i++) y[i] = exp(x[i]);
    }
    Py_DECREF(a);
    return out;
}

static PyObject *check_sum(PyObject *self, PyObject *o)
{
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(o, NPY_FLOAT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!a) return NULL;
    double s = 0.0 + pairwise((const double *)PyArray_DATA(a), PyArray_DIM(a, 0));
    Py_DECREF(a);
    return PyFloat_FromDouble(s);
}

static PyMethodDef methods[] = {
    {"setup", setup, METH_O, "setup(Node): read Node's slot offsets"},
    {"descend", descend, METH_VARARGS, "one descent of v5's selection, as MCTS._descend"},
    {"unwind", unwind, METH_O, "MCTS._unwind"},
    {"backup", backup, METH_VARARGS, "MCTS._backup for power_p 1, proofs and moves-left off"},
    {"expand_rows", expand_rows, METH_VARARGS, "_expand's prior and its five zero rows, or None"},
    {"check_exp", check_exp, METH_O, "exp over a float64 row (the load check)"},
    {"check_sum", check_sum, METH_O, "the pairwise sum of a float64 row (the load check)"},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "mayfly_walk", NULL, -1, methods};
PyMODINIT_FUNC PyInit_mayfly_walk(void)
{
    import_array();
    return PyModule_Create(&module);
}
