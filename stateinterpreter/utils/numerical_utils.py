
import numpy as np
from scipy.optimize import minimize
from .._configs import *
from ._compiled_numerics import _evaluate_kde_args, _evaluate_logkde_args, logsumexp

def cov(points, weights=None):
    assert len(points.shape) == 2
    n_pts, dims = points.shape

    #Sanitize Weights
    if weights is not None:
        assert len(weights.shape) == 1
        assert weights.shape[0] == n_pts
        assert np.abs(np.sum(weights) - 1) <= __EPS__
    else:
        weights = np.ones(n_pts)/n_pts

    mu = np.sum(weights*(points.T), axis=1)
    _centered_pts = np.sqrt(weights)*((points - mu).T) #(dims, n_pts)
    _cov = _centered_pts@(_centered_pts.T)*1/(1 - np.sum(weights**2))
    return _cov 

def weights_from_logweights(logweights):
    """Check if this is somewhat stable"""
    C = logsumexp(logweights)
    return np.exp(logweights - C)

class gaussian_kde:
    def __init__(self, dataset, bandwidth, logweights=None):
        assert len(dataset.shape) == 2
        self.dataset = dataset #[# of pts, # of dims]
        self.n_centers, self.dims = self.dataset.shape
        
        #Sanitize Weights
        if logweights is not None:
            assert len(logweights.shape) == 1
            assert logweights.shape[0] == self.n_centers
        else:
            logweights = np.zeros(self.n_centers)
        self.logweights = logweights
        self._logweights_norm = logsumexp(self.logweights)
        self.weights = weights_from_logweights(self.logweights)

        self._sqrt_inv_cov = np.linalg.cholesky(self.inv_cov)
         self._sqrt_inv_cov_det = np.prod(np.diag(self._sqrt_inv_cov))
         self._sqrt_inv_cov_log_det = np.sum(np.log(np.diag(self._sqrt_inv_cov)))
        
        try:
            bandwidth = np.array(bandwidth)
        except:
            ValueError("Unable to convert bandwidth to np.array")

        if bandwidth.ndim == 0:
            self.bwidth = np.eye(self.dims)*bandwidth
            self._bwidth_norm = self.dims*np.log(bandwidth)
        elif bandwidth.ndim == 1:
            assert bandwidth.shape[0] == self.ndim, "Dimensions of bandwidth vector do not match data dimensions."
            self._bwidth_norm = np.sum(np.log(bandwidth))
            self.bwidth = np.diag(bandwidth)     
        else:
            msg = "`bandwidth` should be scalar, matrix or vector"
            raise ValueError(msg)
    
        self.covariance = cov(dataset, weights=self.weights)
        
    def __call__(self, points, logpdf=False):
        #Evaluate the estimated pdf on a provided set of points.
        res = self._kde_eval(points, logpdf=logpdf)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def logpdf(self, points):
        return self.__call__(points, logpdf=True)
    
    def grad(self, points, logpdf=False):
        return self.__call__(points, logpdf=logpdf, grads=True)

    def _kde_eval(self, points, logpdf=False):
        if (points.ndim ==1):
            assert points.shape[0] == self.dims
            points = points[np.newaxis, :]

        dataset = np.dot(self.dataset, self.bwidth**-1)
        points = np.dot(points, self.bwidth**-1)
        dtype = points.dtype
        if logpdf:
            return _evaluate_logkde_args(points, dataset, self._bwidth_norm, self.logweights, self._logweights_norm, dtype)
        else:
            return _evaluate_kde_args(points, dataset, self._bwidth_norm, self.logweights, self._logweights_norm, dtype)          

    
    def sample(self, size = 1):
        centers = np.random.choice(self.n_centers, size=size, p=self.weights)
        displacements = np.random.multivariate_normal(
            np.zeros((self.dims,)),
            (self.bwidth**2)*self.covariance,
            size=size
        )
        return self.dataset[centers] + displacements

    def local_minima(self, num_init=100, decimals_tolerance=4, sampling='uniform'):
        assert type(num_init) == int, '"num_init" should be an int.'
        if sampling == 'data_driven':
            init_pts = self.sample(size=num_init)
        elif sampling == 'uniform':
            bounds = [((x.max() + x.min())/2, x.max() - x.min()) for x in self.dataset.T]
            init_pts = np.random.rand(self.dims, num_init) - 0.5
            init_pts = np.vstack([
                init_pts[i]*(bounds[i][1]) + bounds[i][0] for i in range(self.dims)
            ]).T
        else:
            raise KeyError(f'Key "{sampling}" not allowed. Valid values: "data_driven","uniform".')     
        objective = lambda x: -self.__call__(x, logpdf=True)
        found_minima = []
        for pt in init_pts:
            res = minimize(objective, pt)
            if res.success:
                found_minima.append(res.x)
        found_minima = np.asarray(found_minima).round(decimals=decimals_tolerance)
        real_minima = np.unique(found_minima, axis=0)
        return real_minima