
import numpy as np
from scipy.optimize import minimize
from ._configs import *
from ._numerics import _evaluate_kde_args, _evaluate_logkde_args, logsumexp

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
    def __init__(self, dataset, bw_method=None, logweights=None):
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

        if bw_method is None:
            self.bwidth = self.scotts_factor()
        elif bw_method == 'scott':
            self.bwidth = self.scotts_factor()
        elif bw_method == 'silverman':
            self.bwidth = self.silverman_factor()
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self.bwidth = bw_method
        else:
            msg = "`bw_method` should be 'scott', 'silverman' or a scalar "
            raise ValueError(msg)
         
        self.covariance = cov(dataset, weights=self.weights)
        self.inv_cov = np.linalg.inv(self.covariance)
        
        self._sqrt_inv_cov = np.linalg.cholesky(self.inv_cov)
        self._sqrt_inv_cov_det = np.prod(np.diag(self._sqrt_inv_cov))
        self._sqrt_inv_cov_log_det = np.sum(np.log(np.diag(self._sqrt_inv_cov)))

        
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

        points = np.dot(points, self._sqrt_inv_cov)
        dataset = np.dot(self.dataset, self._sqrt_inv_cov)
        dtype = points.dtype
        if logpdf:
            return _evaluate_logkde_args(points, dataset, self.bwidth, self.logweights, self._logweights_norm, self._sqrt_inv_cov_log_det, dtype)
        else:
            return _evaluate_kde_args(points, dataset, self.bwidth, self.logweights, self._logweights_norm, self._sqrt_inv_cov_log_det, dtype)          

    @property
    def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1/sum(self.weights**2)
            return self._neff
        
    def scotts_factor(self):
        """Compute Scott's factor.
            +++ From scipy.stats.gaussian_kde +++
            Returns
            -------
            s : float
                Scott's factor.
        """
        return np.power(self.neff, -1./(self.dims+4))

    def silverman_factor(self):
        """Compute the Silverman factor.
            +++ From scipy.stats.gaussian_kde +++
            Returns
            -------
            s : float
                The silverman factor.
        """
        return np.power(self.neff*(self.dims+2.0)/4.0, -1./(self.dims+4))
    
    def sample(self, size = 1):
        centers = np.random.choice(self.n_centers, size=size, p=self.weights)
        displacements = np.random.multivariate_normal(
            np.zeros((self.dims,)),
            (self.bwidth**2)*self.covariance,
            size=size
        )
        return self.dataset[centers] + displacements

    def local_minima(self, num_init=100, decimals_tolerance=4, sampling='data_driven'):
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