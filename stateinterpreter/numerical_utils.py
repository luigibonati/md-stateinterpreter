
import numpy as np
from scipy.special import logsumexp

_EPS = 1e-10


def cov(points, weights=None):
    assert len(points.shape) == 2
    n_pts, dims = points.shape

    #Sanitize Weights
    if weights is not None:
        assert len(weights.shape) == 1
        assert weights.shape[0] == n_pts
        assert np.abs(np.sum(weights) - 1) <= _EPS
    else:
        weights = np.ones(n_pts)/n_pts

    mu = np.sum(weights*(points.T), axis=1)
    _centered_pts = np.sqrt(weights)*((points - mu).T) #(dims, n_pts)
    _cov = _centered_pts@(_centered_pts.T)*1/(1 - np.sum(weights**2))
    return _cov 



class gaussian_kde:
    #Trying to implement SciPy api
    def __init__(self, dataset, bw_method=None, weights=None):
        assert len(dataset.shape) == 2
        self.dataset = dataset #(# of dims, # of data)
        self.n_pts, self.dims = self.dataset.shape 
        #Sanitize Weights
        if weights is not None:
            assert len(weights.shape) == 1
            assert weights.shape[0] == self.n_pts
            assert np.abs(np.sum(weights) - 1) <= _EPS
        else:
            weights = np.ones(self.n_pts)/self.n_pts
        self.weights = weights

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

        self.covariance = cov(dataset, weights=weights)
        self.inv_cov = np.linalg.inv(self.covariance)
        
        self._sqrt_cov = np.diag(np.linalg.cholesky(self.covariance))
        self._sqrt_cov_det = np.prod(self._sqrt_cov)
        self._sqrt_cov_log_det = np.sum(np.log(self._sqrt_cov))

        
    def __call__(self, points):
        #Evaluate the estimated pdf on a provided set of points.
        norm = (np.power(2*np.pi*(self.bwidth**2),0.5*self.dims)*self._sqrt_cov_det)**-1
        #Not the fastest option, but at least it works
        X = (self.dataset[np.newaxis,...] - points[:,np.newaxis,:]) #[n_points, n_dataset, dims]
        arg = -np.sum(np.dot(X, self.inv_cov)*X, axis=-1) #[n_points, n_dataset]
        arg /= 2*(self.bwidth**2)
        kde = np.exp(arg)
        return np.dot(kde, self.weights)*norm

    def logpdf(self, points):
        #Evaluate the estimated logpdf on a provided set of points.
        
        X = (self.dataset[np.newaxis,...] - points[:,np.newaxis,:]) #[n_points, n_dataset, dims]
        arg = -np.sum(np.dot(X, self.inv_cov)*X, axis=-1) #[n_points, n_dataset]
        arg /= 2*(self.bwidth**2)
        return logsumexp(arg.T + np.log(self.weights)[:,np.newaxis] -0.5*self.dims*np.log(2*np.pi*(self.bwidth**2)) - self._sqrt_cov_log_det, axis = 0)

    def grad(points):
        #Evaluate the estimated pdf grad on a provided set of points.
        raise NotImplementedError("Not yet implemented")

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