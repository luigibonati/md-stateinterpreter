
import numpy as np
import concurrent.futures
from numba import jit, prange
from scipy.optimize import minimize
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from os import cpu_count
from stateinterpreter._numerics import _evaluate_kde_args_cython, _evaluate_logkde_args_cython

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

def weights_from_logweights(logweights):
    """Check if this is somewhat stable"""
    C = logsumexp(logweights)
    return np.exp(logweights - C)

@jit(nopython=True, parallel=False)
def _evaluate_kde_args(logpdf, points, dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det):
    args = np.zeros((points.shape[0]))
    for idx in prange(points.shape[0]):
        args[idx] += _evaluate_one_arg(logpdf, points[idx], dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det)
    return args

def _evaluate_kde_grads(logpdf, points, dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det):
    grads = np.zeros_like(points)
    for idx in prange(points.shape[0]):
        grads[idx] += _evaluate_one_grad(logpdf, points[idx], dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det)
    return grads

@jit(nopython=True, fastmath=True)
def logsumexp(X):
    alpha = -np.Inf
    r = 0.0
    for x in X:
        if x != -np.Inf:
            if x <= alpha:
                r += np.exp(x - alpha)
            else:
                r *= np.exp(alpha - x)
                r += 1.0
                alpha = x
    return np.log(r) + alpha

@jit(nopython=True)
def _evaluate_one_arg(logpdf, pt, dataset, inv_cov, bwidth, logweights, logweights_norm, sqrt_cov_log_det):
    dims = dataset.shape[1]
    X = dataset - pt
    arg = -np.sum(np.dot(X, inv_cov)*X, axis=-1) #[n_centers]
    arg /= 2*(bwidth**2)
    arg += logweights - logweights_norm -0.5*dims*np.log(2*np.pi*(bwidth**2)) - sqrt_cov_log_det
    if logpdf:
        return logsumexp(arg)
    else:
        return np.sum(np.exp(arg))

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

class gaussian_kde:
    #Trying to implement SciPy api
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
        
        self._sqrt_cov = np.linalg.cholesky(self.inv_cov)
        self._sqrt_cov_det = np.prod(np.diag(self._sqrt_cov))
        self._sqrt_cov_log_det = np.sum(np.log(np.diag(self._sqrt_cov)))

        
    def __call__(self, points, logpdf=False, grads=False, cython=True):
        #Evaluate the estimated pdf on a provided set of points.
        res = self._kde_eval(points, logpdf=logpdf, grads=grads, cython=cython)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def logpdf(self, points):
        return self.__call__(points, logpdf=True)
    
    def grad(self, points, logpdf=False):
        return self.__call__(points, logpdf=logpdf, grads=True)

    def _kde_eval(self, points, logpdf=False, grads=False, cython=True):
        if (points.ndim ==1):
            assert points.shape[0] == self.dims
            points = points[np.newaxis, :]

        if grads:
            return _evaluate_kde_grads(logpdf, points, self.dataset, self.inv_cov, self.bwidth, self.logweights, self._logweights_norm, self._sqrt_cov_log_det)
        else:
            if cython:
                points = np.dot(points, self._sqrt_cov)
                dataset = np.dot(self.dataset, self._sqrt_cov)
                dtype = points.dtype
                if logpdf:
                    return _evaluate_logkde_args_cython(points, dataset, self.bwidth, self.logweights, self._logweights_norm, self._sqrt_cov_log_det, dtype)
                else:
                    return _evaluate_kde_args_cython(points, dataset, self.bwidth, self.logweights, self._logweights_norm, self._sqrt_cov_log_det, dtype)
            else:
                return _evaluate_kde_args(logpdf, points, self.dataset, self.inv_cov, self.bwidth, self.logweights, self._logweights_norm, self._sqrt_cov_log_det)

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


def local_minima(objective, bounds, method=None, method_kwargs=dict()):

    assert callable(objective), "The given objective function should be a callable"
    if method is None:
        method = _brute_rand_minima
    elif method == 'brute':
        method = _brute_grid_minima
    elif method == 'rand_brute':
        method = _brute_rand_minima
    else:
        msg = "`method` should be 'brute' or 'rand_brute' "
        raise ValueError(msg)

    return method(objective, bounds, **method_kwargs)

def _brute_grid_minima(objective, bounds, num_splits=100):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    ndims = len(bounds)
    
    _1d_samples = [np.linspace(vmin, vmax, num_splits) for (vmin, vmax) in bounds]
    meshgrids = np.meshgrid(*_1d_samples)
    sampled_positions = np.array([np.ravel(coord) for coord in meshgrids]).T
    
    f = np.reshape(objective(sampled_positions), (num_splits,)*ndims)
    
    # define an connected neighborhood http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = generate_binary_structure(f.ndim, 2)
    # apply the local minimum filter; all locations of minimum value in their neighborhood are set to 1 http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = minimum_filter(f, footprint=neighborhood) == f
    # local_min is a mask that contains the peaks we are looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask. we create the mask of the background
    background = f == 0
    #
    # a little technicality: we must erode the background in order to successfully subtract it from local_min, otherwise a line will appear along the background border (artifact of the local minimum filter) http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    #
    # we obtain the final mask, containing only peaks, by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    detected_minima = np.where(detected_minima)
    presumed_minima = []

    for idxs in zip(*detected_minima):
        minima = []
        for coord in meshgrids:
            minima.append(coord[idxs])
        presumed_minima.append(np.array(minima))
    presumed_minima = np.array(presumed_minima)

    real_minima = []
    for minima in presumed_minima:

        res = minimize(objective, minima)
        if res.success:
            real_minima.append(res.x)
        else:
            real_minima.append(minima)
    real_minima = np.array(real_minima)
    f_min =np.asarray([objective(x) for x in real_minima])
    sortperm = np.argsort(f_min, axis=0)
    return real_minima[sortperm]

def _brute_rand_minima(objective, bounds, num_init=100, decimals_tolerance=4):
    init_pts = np.random.rand(num_init, len(bounds)) - 0.5 #centering
    for idx, bd in enumerate(bounds):
        vmin, vmax = bd
        mean = 0.5*(vmax + vmin)
        scale = vmax - vmin
        init_pts[:,idx] *= scale
        init_pts[:, idx] += mean
    chunks = np.array_split(init_pts, cpu_count(), axis=0)

    found_minima = []

    def _minimize_chunk(chunk):
        minima = []
        for pt in chunk:
            res = minimize(objective, pt)
            if res.success:
                minima.append(res.x)
        return minima

    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
            fut = [executor.submit(_minimize_chunk, chunk) for chunk in chunks]
            for fut_result in concurrent.futures.as_completed(fut):
                _chunk_minima = fut_result.result()
                found_minima.extend(_chunk_minima)
    """
    for chunk in chunks:
        found_minima.extend(_minimize_chunk(chunk))
    found_minima = np.asarray(found_minima).round(decimals=decimals_tolerance)
    real_minima = np.unique(found_minima, axis=0)
    f_min =np.asarray([objective(x) for x in real_minima])
    sortperm = np.argsort(f_min, axis=0)
    return real_minima[sortperm]


  



