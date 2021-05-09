#cython: language_level=3
#cython: initializedcheck=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True

import cython
from libc.math cimport sqrt, exp, pi, frexp, exp2, exp, log
cimport numpy as cnp
import numpy as np

cdef inline double _psi(int n, double x) nogil:
    cdef double h, h_prev, sum_log_scale, scale
    cdef int i, log_scale
    h_prev = pi ** -0.25
    h = sqrt(2.0) * x * pi ** -0.25
    sum_log_scale = 0
    if n == 0:
        h = h_prev
    for i in range(2, n + 1):
        h, h_prev = sqrt(2.0 / i) * x * h - sqrt(float(i - 1) / i) * h_prev, h
        frexp(h, &log_scale)
        scale = exp2(-log_scale)
        h *= scale
        h_prev *= scale
        sum_log_scale += log_scale
    return h * exp(-(x ** 2) / 2 + log(2) * sum_log_scale)


def psi(int n, x):
    cdef int i
    if isinstance(x, np.ndarray) and x.ndim:
        out = np.empty(x.shape)
        for i in range(x.size):
            out.flat[i] = _psi(n, x.flat[i])
        return out
    return _psi(n, x)
    
