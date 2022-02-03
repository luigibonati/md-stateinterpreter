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
cpdef GaussianKDE_pdf(real[:,:] points, real[:,:] kde_centers, real bwidth_norm, real[:] logweights, real _logweights_norm, dtype):
    cdef int data_dim = kde_centers.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i, j
    cdef real[:] result = np.zeros(x_dim, dtype)
    cdef double norm = _logweights_norm + 0.5*d*math.log(2*PI) - bwidth_norm
    cdef double res
    for i in range(x_dim):
        res = 0
        for j in prange(data_dim, nogil=True):
            res += math.exp(KDE_of_point_i_center_j(i, j, points, kde_centers, bwidth_norm, norm, logweights[j], d))
        result[i] = res
    return np.asarray(result)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef GaussianKDE_grad(real[:,:] points, real[:,:] kde_centers, real bwidth_norm, real[:] logweights, real _logweights_norm, dtype):
    cdef int data_dim = kde_centers.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i, j, k
    cdef real[:,:] result = np.zeros((x_dim, d), dtype)
    cdef double norm = _logweights_norm + 0.5*d*math.log(2*PI) - bwidth_norm
    cdef double res
    for i in range(x_dim):
        for k in range(d):
            res = 0
            for j in prange(data_dim, nogil=True):
                res -= math.exp(KDE_of_point_i_center_j(i, j, points, kde_centers, bwidth_norm, norm, logweights[j], d))*(points[i,k] - kde_centers[j,k])  
            result[i,k] = res
    return np.asarray(result)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef GaussianKDE_loggrad(real[:,:] points, real[:,:] kde_centers, real bwidth_norm, real[:] logweights, real _logweights_norm, dtype):
    cdef int data_dim = kde_centers.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i, j, k
    cdef real[:,:] result = np.zeros((x_dim, d), dtype)
    cdef double norm = _logweights_norm + 0.5*d*math.log(2*PI) - bwidth_norm
    cdef double res, tmp_max, tmp_sum
    cdef real[:] tmp_storage = np.empty(data_dim)
    for i in range(x_dim): 
        res = 0
        tmp_max = -INFINITY
        tmp_sum = 0
        for j in range(data_dim):
            tmp_storage[j] = KDE_of_point_i_center_j(i, j, points, kde_centers, bwidth_norm, norm, logweights[j], d)
            if tmp_max < tmp_storage[j]:
                tmp_max = tmp_storage[j]
        for j in prange(data_dim, nogil=True):
            tmp_sum += math.exp(tmp_storage[j] - tmp_max)
        for k in range(d):
            res = 0
            for j in prange(data_dim, nogil=True):
                res -= math.exp(tmp_storage[j] - tmp_max)*(points[i,k] - kde_centers[j,k])/tmp_sum
            result[i,k] = res
    return np.asarray(result)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef real KDE_of_point_i_center_j(Py_ssize_t i, Py_ssize_t j, real[:,:] points, real[:,:] kde_centers, real bwidth_norm, real norm, real logweight, int d) nogil:
    cdef Py_ssize_t k  
    cdef real arg = 0
    for k in range(d):
        arg += (points[i,k] - kde_centers[j,k])*(points[i,k] - kde_centers[j,k])
    arg = -arg/2 - norm + logweight
    return arg

@cython.boundscheck(False)
@cython.wraparound(False)
def GaussianKDE_logpdf(real[:,:] points, real[:,:] kde_centers, real bwidth_norm, real[:] logweights, real _logweights_norm, dtype):
    cdef int data_dim = kde_centers.shape[0]
    cdef int x_dim = points.shape[0]
    cdef int d = points.shape[1]
    cdef Py_ssize_t i
    cdef real[:] result = np.zeros(x_dim, dtype)
    cdef double norm = _logweights_norm + 0.5*d*np.log(2*PI) - bwidth_norm
    for i in prange(x_dim, nogil=True):
        result[i] = logpdf_point_i(i, points, kde_centers, bwidth_norm, norm, logweights, d, data_dim)
    return np.asarray(result)  

@cython.boundscheck(False)
@cython.wraparound(False)
cdef real logpdf_point_i(Py_ssize_t i, real[:,:] points, real[:,:] kde_centers , real bwidth_norm, real norm, real[:] logweights, int d, int data_dim) nogil:
    cdef real alpha = -INFINITY
    cdef real r = 0
    cdef real arg
    cdef Py_ssize_t j
    for j in range(data_dim):
        arg = KDE_of_point_i_center_j(i, j, points, kde_centers, bwidth_norm, norm, logweights[j], d)
        if arg != - INFINITY:
            if arg <= alpha:
                r += math.exp(arg - alpha)
            else:
                r *= math.exp(alpha - arg)
                r += 1
                alpha = arg
    return math.log(r) + alpha