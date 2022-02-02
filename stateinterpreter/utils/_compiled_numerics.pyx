
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
def contact_function(real[:,:] x, real r0=1.0, real d0=0, int n=6, int m=12):
    cdef float[:,:] result = np.zeros_like(x, dtype=np.single)
    cdef Py_ssize_t i, j
    for i in prange(x.shape[1], nogil=True):
        for j in range(x.shape[0]):
            result[j, i] += _contact(x, i, j, r0, d0, n, m)
    return np.asarray(result)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float _contact(real[:,:] x, Py_ssize_t i, Py_ssize_t j, real r0, real d0, int n, int m) nogil:
    cdef float y = (x[j,i] - d0) / r0
    # (see formula for RATIONAL) https://www.plumed.org/doc-v2.6/user-doc/html/switchingfunction.html
    if y == 1:
        #Handling limiting case
        return n/m
    else:
        return (1 - math.pow(y, n))/(1 - math.pow(y, m))

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
cpdef _evaluate_kde_args(real[:,:] points, real[:,:] dataset, real bwidth, real[:] logweights, real _logweights_norm, dtype):
    cdef int data_dim = dataset.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i, j
    cdef real[:] result = np.zeros(x_dim, dtype)
    cdef double norm = _logweights_norm + 0.5*d*math.log(2*PI*(bwidth*bwidth))
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
def _evaluate_logkde_args(real[:,:] points, real[:,:] dataset, real bwidth, real[:] logweights, real _logweights_norm, dtype):
    cdef int data_dim = dataset.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i, j
    cdef real[:] result = np.zeros(x_dim, dtype)
    cdef double norm = _logweights_norm + 0.5*d*np.log(2*PI*(bwidth*bwidth))
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


'''
def _evaluate_kde_grads(logpdf, points, dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det):
grads = np.zeros_like(points)
for idx in range(points.shape[0]):
    grads[idx] += _evaluate_one_grad(logpdf, points[idx], dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det)
return grads



@jit(nopython=True)
def _evaluate_one_grad(logpdf, pt, dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det):
    dims = dataset.shape[1]
    X = dataset - pt
    arg = -np.sum(np.dot(X, inv_cov)*X, axis=-1) #[n_centers]
    arg /= 2*(bwidth**2)
    arg += logweights - logweights_norm -0.5*dims*np.log(2*np.pi*(bwidth**2)) - sqrt_cov_log_det
    grad_pdf = np.sum(np.exp(arg)*np.dot(X, inv_cov).T, axis=1)
    if logpdf:
        return grad_pdf/np.sum(np.exp(arg))
    else:
        return grad_pdf
'''