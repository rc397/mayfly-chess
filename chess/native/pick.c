/* The arithmetic of mcts.MCTS._pick_fast in C, the same float32 numbers bit for bit.

   The selection v4 and v5 play (q_init, puct_v) spends most of the tree's CPU in about twenty-five numpy
   calls a pick. This does the same arithmetic in one pass. It is a speedup that may reach sealed versions
   only because it changes no number. Every operation below is the numpy one, in the same order, each
   rounded to float32, and harness.py proves it (stub and real network, v1 to v4 and the v5 defaults).

   What makes it exact.
   - A Python float meeting a float32 array is cast to float32 first (NumPy 2's weak scalars, NEP 50), so
     every scalar arrives here as a double and is cast with (float), round to nearest.
   - float arithmetic on x86-64 is SSE, one rounding per operation, and nothing may be contracted into a
     fused multiply-add. MSVC's /fp:precise does not contract (setup.py), GCC needs -ffp-contract=off.
   - sqrtf is IEEE correctly rounded, like np.sqrt on float32. np.maximum(t, 0) keeps NaN and -0.0, and so
     does (t < 0 ? 0 : t).
   - np.argmax returns the first NaN if there is one, otherwise the first maximum.

   The forcing branch, the q_floor and every other selection stay in Python (_pick_fast falls back). */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static int get_f32(PyObject *o, Py_buffer *b, Py_ssize_t n)
{
    if (PyObject_GetBuffer(o, b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0)
        return -1;
    if (b->itemsize != 4 || b->format == NULL || b->format[0] != 'f' || b->format[1] != '\0'
        || b->ndim != 1 || (n >= 0 && b->shape[0] != n)) {
        PyBuffer_Release(b);
        PyErr_SetString(PyExc_TypeError, "pick: needs 1-D contiguous float32 arrays of one length");
        return -1;
    }
    return 0;
}

/* pick(N, W, P, Q0, V2, n0, c1, c2x2, sqrtN, logN1, grow) -> (edge, q of that edge)
   grow <= 0 means explore_growth is off (the factor is not applied at all, as in Python). */
static PyObject *pick(PyObject *self, PyObject *args)
{
    PyObject *oN, *oW, *oP, *oQ, *oV;
    double n0, c1, c2x2, sqrtN, logN1, grow;
    if (!PyArg_ParseTuple(args, "OOOOOdddddd", &oN, &oW, &oP, &oQ, &oV, &n0, &c1, &c2x2, &sqrtN, &logN1, &grow))
        return NULL;
    Py_buffer bN, bW, bP, bQ, bV;
    if (get_f32(oN, &bN, -1) != 0)
        return NULL;
    Py_ssize_t k = bN.shape[0];
    if (get_f32(oW, &bW, k) != 0) { PyBuffer_Release(&bN); return NULL; }
    if (get_f32(oP, &bP, k) != 0) { PyBuffer_Release(&bN); PyBuffer_Release(&bW); return NULL; }
    if (get_f32(oQ, &bQ, k) != 0) { PyBuffer_Release(&bN); PyBuffer_Release(&bW); PyBuffer_Release(&bP); return NULL; }
    if (get_f32(oV, &bV, k) != 0) {
        PyBuffer_Release(&bN); PyBuffer_Release(&bW); PyBuffer_Release(&bP); PyBuffer_Release(&bQ); return NULL;
    }
    const float *n = (const float *)bN.buf, *w = (const float *)bW.buf, *p = (const float *)bP.buf;
    const float *q0 = (const float *)bQ.buf, *v2 = (const float *)bV.buf;
    const float n0f = (float)n0, c1f = (float)c1, c2f = (float)c2x2;
    const float sqf = (float)sqrtN, lgf = (float)logN1, grf = (float)grow;
    const int growing = grow > 0.0;

    Py_ssize_t best = -1, first_nan = -1;
    float bestu = 0.0f, bestq = 0.0f, nanq = 0.0f;
    for (Py_ssize_t i = 0; i < k; i++) {
        /* q = (n0 * q0 + w) / (n0 + n) */
        float a = n0f * q0[i];
        a = a + w[i];
        float den = n0f + n[i];
        float q = a / den;
        /* the spread over visited edges, 1 for an unvisited one */
        float sd;
        if (n[i] > 0.0f) {
            float mean = w[i] / n[i];
            float t = v2[i] / n[i];
            float mm = mean * mean;
            t = t - mm;
            t = (t < 0.0f) ? 0.0f : t;
            sd = sqrtf(t);
        } else {
            sd = 1.0f;
        }
        /* u = (c1 * p * sd * sqrt(N)) / d + (2 c2 * p * log(N + 1)) / d, d = 1 + n */
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
        if (u != u) {                       /* NaN, and numpy's argmax returns the first one */
            if (first_nan < 0) { first_nan = i; nanq = q; }
            continue;
        }
        if (best < 0 || u > bestu) { best = i; bestu = u; bestq = q; }
    }
    PyBuffer_Release(&bN); PyBuffer_Release(&bW); PyBuffer_Release(&bP); PyBuffer_Release(&bQ); PyBuffer_Release(&bV);
    if (first_nan >= 0)
        return Py_BuildValue("nd", first_nan, (double)nanq);
    if (best < 0) {
        PyErr_SetString(PyExc_ValueError, "pick: an empty node");
        return NULL;
    }
    return Py_BuildValue("nd", best, (double)bestq);
}

static PyMethodDef methods[] = {
    {"pick", pick, METH_VARARGS, "The q_init + puct_v pick of _pick_fast: (edge, q)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "mayfly_pick", NULL, -1, methods};

PyMODINIT_FUNC PyInit_mayfly_pick(void) { return PyModule_Create(&module); }
