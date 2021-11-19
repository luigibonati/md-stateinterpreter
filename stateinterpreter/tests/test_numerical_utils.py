"""
Unit and regression test for the stateinterpreter package.
"""

# Import package, test suite, and other packages as needed
from scipy.stats import gaussian_kde as scipy_gaussian_kde
from stateinterpreter.numerical_utils import gaussian_kde
import numpy as np

def test_gaussian_kde():
    """Sample test, will always pass so long as import statement worked."""
    n_centers = 1000
    n_dims = 3
    n_pts = 1000
    rand_dataset = np.random.rand(n_centers, n_dims)
    rand_points = np.random.rand(n_pts, n_dims)

    #Test 
    scipy_KDE = scipy_gaussian_kde(rand_dataset.T)
    KDE = gaussian_kde(rand_dataset)

    d_pdf = np.max(np.abs(scipy_KDE(rand_points.T) - KDE(rand_points)))
    d_logpdf = np.max(np.abs(scipy_KDE.logpdf(rand_points.T) - KDE.logpdf(rand_points)))
    assert d_logpdf  < 1e-10, f"logpdf delta is {d_logpdf}"
    assert d_pdf  < 1e-10, f"logpdf delta is {d_pdf}"

if __name__ == "__main__":
    print('>>>>> test_gaussian_kde')
    test_gaussian_kde()