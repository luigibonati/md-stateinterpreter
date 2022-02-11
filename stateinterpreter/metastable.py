import pandas as pd
import numpy as np
from .utils.numerical_utils import gaussian_kde
from ._configs import *
import sys

__all__ = ["identify_metastable_states", "approximate_FES"]

def identify_metastable_states(
        colvar,
        selected_cvs,
        kBT,
        bandwidth,
        logweights=None,
        fes_cutoff=None,
        gradient_descent_iterates = 0,
        sort_minima_by = 'cvs_grid',
        optimizer_kwargs=dict(),
    ):
        """Label configurations based on free energy

        Parameters
        ----------
        colvar : Pandas dataframe
            ###
        selected_cvs : list of strings
            Names of the collective variables used for clustering
        kBT : scalar
            Temperature
        bandwidth : scalar
            Bandwidth method for FES calculations
        logweights : pandas.DataFrame, np.array or string , optional
            Logweights used for FES calculation, by default None
        fes_cutoff : float, optional
            Cutoff used to select only low free-energy configurations, if None fes_cutoff is 2k_BT
        sort_minima_by : string, optional
            Sort labels based on `energy`, `cvs`, or `cvs_grid` values, by default `cvs_grid`.
        optimizer_kwargs : optional
            Arguments for optimizer, by default dict(). Possible kwargs are:
                (int) num_init: number of initialization point, 
                (int) decimals_tolerance: number of decimals to retain to identify unique minima,
                (str) sampling: sampling scheme. Accepted strings are 'data_driven' or 'uniform'.
        """

        #Adimensional fes_cutoff
        if fes_cutoff is None:
            fes_cutoff = 2 #kBT
        else:
            fes_cutoff=fes_cutoff/kBT

        # Retrieve logweights
        w = _sanitize_logweights(logweights, colvar=colvar, kBT=kBT)

        # Compute KDE
        empirical_centers = colvar[selected_cvs].to_numpy()
        KDE = gaussian_kde(empirical_centers,bandwidth=bandwidth,logweights=w)

        if __DEV__:
            print("DEV >>> Finding Local Minima") 
        minima = KDE.local_minima(**optimizer_kwargs)
        
        # sort minima based on first CV
        if sort_minima_by == 'energy':
            f_min = np.asarray([-KDE.logpdf(x) for x in minima])
            sortperm = np.argsort(f_min, axis=0)
            minima = minima[sortperm]
        elif sort_minima_by == 'cvs' :
            # sort first by 1st cv, then 2nd, ...
            x = minima
            minima = x [ np.lexsort( np.round(np.flipud(x.T),2) ) ] 
        elif sort_minima_by == 'cvs_grid' :
            bounds = [(x.min(), x.max()) for x in KDE.dataset.T]
            # sort based on a binning of the cvs (10 bins per each direction),
            # along 1st cv, then 2nd, ... 
            x = minima
            y = (x - [ bound[0] for bound in bounds ]) 
            y /= np.asarray( [ bound[1]-bound[0] for bound in bounds ]) / 10
            minima = x [ np.lexsort( np.round(np.flipud(y.T),0) ) ] 
        else:
            raise KeyError(f'Key {sort_minima_by} not allowed. Valid values: "energy","cvs","cvs_grid".')

        # Assign basins and select based on FES cutoff
        basins = _basin_selection(KDE, minima, fes_cutoff, gradient_descent_iterates)

        n_basins = len(basins['labels'].unique())
        print(f"Found {n_basins} local minima with selected populations:")
        for idx in range(n_basins):
            l = len(basins.loc[ (basins['labels'] == idx) & (basins['selection'] == True)])
            print(f"\tBasin {idx} -> {l} configurations.")
        return basins

def _basin_selection(
    KDE, minima, fes_cutoff, gradient_descent_iterates
):
    if __DEV__:
        print("DEV >>> Basin Assignment")
    pts = np.copy(KDE.dataset)
    v = np.zeros_like(pts)
    beta = 0.9 #Default
    learning_rate = np.diag(np.diag(KDE.inv_bwidth)**-1)*0.5
    for _ in range(gradient_descent_iterates): 
        v *= beta
        v += -(1 - beta)*KDE.grad(pts, logpdf=True)
        pts -= np.dot(v,learning_rate)
    norms = np.linalg.norm((pts[:,np.newaxis,:] - minima), axis=2)
    classes = np.argmin(norms, axis=1)
    fes_at_minima = - KDE.logpdf(minima)
    if len(minima) == 1:
        ref_fes = fes_at_minima
    else:
        ref_fes = np.asarray([fes_at_minima[idx] for idx in classes])
    fes_pts = - KDE.logpdf(KDE.dataset)
    mask = (fes_pts - ref_fes) < fes_cutoff
    df = pd.DataFrame(data=classes, columns=["labels"])
    df["selection"] = mask
    return df

def approximate_FES(
        colvar, bandwidth, selected_cvs=None, kBT=2.5, logweights=None
    ):
    """Approximate Free Energy Surface (FES) in the space of selected_cvs through Gaussian Kernel Density Estimation

    Args:
        bandwidth (scalar, vector or matrix):
        selected_cvs (numpy.ndarray or pd.Dataframe): List of sampled collective variables with dimensions [num_timesteps, num_CVs]
        kBT (scalar): Temperature
        logweights (arraylike log weights, optional): Logarithm of the weights. Defaults to None (uniform weights).

    Returns:
        function: Function approximating the free Energy Surface
    """
    if __DEV__:
        print("DEV >>> Approximating FES")
    w = _sanitize_logweights(logweights, colvar=colvar, kBT=kBT)
        
    empirical_centers = colvar[selected_cvs] if selected_cvs is not None else colvar
    empirical_centers = empirical_centers.to_numpy()
    KDE = gaussian_kde(empirical_centers, bandwidth,logweights=w)
    return lambda x: -kBT*KDE.logpdf(x)

def _sanitize_logweights(logweights, colvar=None, kBT=None):
    if logweights is None:
            w = None
            if colvar is not None:
                if ".bias" in colvar.columns:
                    print(
                        "WARNING: a field with .bias is present in colvar, but it is not used for the FES.",file=sys.stderr
                    )
    else:
        if isinstance(logweights, str):
            #Luigi do you think is ok to have multiple behaviours when loading logweights?
            if colvar is None:
                raise ValueError("colvar must be not None, when loading logweights from colvar file")
            if kBT is None:
                raise ValueError("kBT must be not None when loading logweights from colvar file")
            if "*" in logweights:
                w = colvar.filter(regex=logweights.replace('*','')).sum(axis=1).values / kBT
            else:
                w = colvar[logweights].values / kBT
        elif isinstance(logweights, pd.DataFrame):
            w = logweights.values
        elif isinstance(logweights, np.ndarray):
            w = logweights
        else:
            raise TypeError(
                f"{logweights}: Accepted types are 'pandas.Dataframe', 'str' or 'numpy.ndarray' "
            )
        if w.ndim != 1:
            raise ValueError(f"{logweights}: 1D array is required for logweights")
    return w

