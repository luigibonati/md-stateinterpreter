
# cython: infer_types=True

from libc cimport math
cimport cython
from numpy.math cimport INFINITY
from numpy.math cimport PI
import numpy as np
from cython.parallel cimport prange


ctypedef fused real:
    float
    double
    long double

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logsumexp(real[:] X):
    cdef double alpha = -INFINITY
    cdef double r = 0.0
    cdef int n = X.shape[0]
    for i in range(n):
        if X[i] != -INFINITY:
            if X[i] <= alpha:
                r += math.exp(X[i] - alpha)
            else:
                r *= math.exp(alpha - X[i])
                r += 1.0
                alpha = X[i]
    return math.log(r) + alpha

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _evaluate_kde_args_cython(real[:,:] points, real[:,:] dataset, real bwidth, real[:] logweights, real _logweights_norm, real _sqrt_cov_log_det, dtype):
    cdef int data_dim = dataset.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i, j
    cdef real[:] result = np.zeros(x_dim, dtype)
    cdef double norm = _logweights_norm + 0.5*d*math.log(2*PI*(bwidth*bwidth)) + _sqrt_cov_log_det
    cdef double res
    for i in range(x_dim):
        res = 0
        for j in prange(data_dim, nogil=True):
            res += math.exp(_get_arg(i, j, points, dataset, bwidth, norm, logweights[j], d))
        result[i] = res
    return np.asarray(result)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef real _get_arg(Py_ssize_t i, Py_ssize_t j, real[:,:] points, real[:,:] dataset, real bwidth, real norm, real logweight, int d) nogil:
    cdef Py_ssize_t k  
    cdef real arg = 0
    for k in range(d):
        arg += (points[i,k] - dataset[j,k])*(points[i,k] - dataset[j,k])
    arg = -arg/(2*bwidth*bwidth) - norm + logweight
    return arg


@cython.boundscheck(False)
@cython.wraparound(False)
def _evaluate_logkde_args_cython(real[:,:] points, real[:,:] dataset, real bwidth, real[:] logweights, real _logweights_norm, real _sqrt_cov_log_det, dtype):
    cdef int data_dim = dataset.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i, j
    cdef real[:] result = np.zeros(x_dim, dtype)
    cdef double norm = _logweights_norm + 0.5*d*np.log(2*PI*(bwidth*bwidth)) + _sqrt_cov_log_det
    cdef double bwidth_ = bwidth
    for i in prange(x_dim, nogil=True):
        result[i] = _get_log_arg(i, points, dataset , bwidth, norm, logweights, d, data_dim)
    return np.asarray(result)  

@cython.boundscheck(False)
@cython.wraparound(False)
cdef real _get_log_arg(Py_ssize_t i, real[:,:] points, real[:,:] dataset , real bwidth, real norm, real[:] logweights, int d, int data_dim) nogil:
    cdef real alpha = -INFINITY
    cdef real r = 0
    cdef real arg
    cdef Py_ssize_t j
    for j in range(data_dim):
        arg = _get_arg(i, j, points, dataset, bwidth, norm, logweights[j], d)
        if arg != - INFINITY:
            if arg <= alpha:
                r += math.exp(arg - alpha)
            else:
                r *= math.exp(alpha - arg)
                r += 1
                alpha = arg
    return math.log(r) + alpha