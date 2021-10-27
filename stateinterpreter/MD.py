# imports

import sys
import numpy as np
import pandas as pd
import mdtraj as md
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm


from .io import load_dataframe
from .numerical_utils import gaussian_kde, local_minima

__all__ = ["Loader"]

"""
OUTLINE
=======
1a. Load collective variables
   - from FILE or pd.DataFrame
1b. Load descriptors (optional)
   - from FILE or pd.DataFrame
2. (optional: load trajectory and compute descriptors)
3. Identify states from FES
4. Get dataframe (CVs, descriptors, labels)
"""


class Loader:
    def __init__(
        self, colvar, descriptors=None, kbt=2.5, start=0, stride=1, _DEV=False, **kwargs
    ):
        """Prepare inputs for stateinterpreter

        Parameters
        ----------
        colvar : pandas.DataFrame or string
            collective variables
        descriptors : pandas.DataFrame or string, optional
            input features, by default None
        kbt : float, optional
            temperature [KbT], by default 2.5
        start : int, optional
            keep data from value, by default 0
        stride : int, optional
            keep data every stride, by default 1
        _DEV : bool, optional
            enable debug mode, by default False

        Examples
        --------
        Load collective variables and descriptors from file, and store them as DataFrames

        >>> folder = 'stateinterpreter/data/test-chignolin/'
        >>> colvar_file = folder + 'COLVAR'
        >>> descr_file = folder+ 'DESCRIPTORS.csv'
        >>> data = Loader(colvar_file, descr_file, kbt=2.8, stride=10)
        >>> print(f"Colvar: {data.colvar.shape}, Descriptors: {data.descriptors.shape}")
        Colvar: (105, 9), Descriptors: (105, 783)

        """
        # collective variables data
        self.colvar = load_dataframe(colvar, **kwargs)
        self.colvar = self.colvar.iloc[start::stride, :]
        if ( stride > 1 ) or ( start > 0 ):
            self.colvar.reset_index(drop=True, inplace=True)
        if _DEV:
            print(f"Collective variables: {self.colvar.values.shape}")

        # descriptors data
        if descriptors is not None:
            self.descriptors = load_dataframe(descriptors, **kwargs)
            self.descriptors = self.descriptors.iloc[start::stride, :]
            if "time" in self.descriptors.columns:
                self.descriptors = self.descriptors.drop("time", axis="columns")
            if _DEV:
                print(f"Descriptors: {self.descriptors.shape}")
            assert len(self.colvar) == len(
                self.descriptors
            ), "mismatch between colvar and descriptor length."

        # save attributes
        self.kbt = kbt
        self.stride = stride
        self.start = start
        self._DEV = _DEV

        # initialize attributes to None
        self.traj = None
        self.basins = None
        self.n_basins = None

    def load_trajectory(self, traj_dict, descriptors = ['ca','dihedrals','hbonds']):
        """ "Load trajectory with mdtraj.

        Parameters
        ----------
        traj_dict : dict
            dictionary containing trajectory and topology (optional) file
        descriptors : bool or list, default True
            compute list of descriptors. if True compute all descriptors: ['ca','dihedral','hbonds'].
            if False no descriptor is computed.

        """

        traj_file = traj_dict["trajectory"]
        topo_file = traj_dict["topology"] if "topology" in traj_dict else None

        self.traj = md.load(traj_file, top=topo_file, stride=self.stride)
        if self.start > 0:
            self.traj = self.traj[int(self.start/self.stride) : ]

        assert len(self.traj) == len(
            self.colvar
        ), f"length traj ({len(self.traj)}) != length colvar ({len(self.colvar)})"

        # Compute descriptors
        if descriptors is True: # if true compute all of them
            self.compute_descriptors()
        elif descriptors is not False:
            self.compute_descriptors(descriptors)

    def compute_descriptors(self,descriptors=['ca','dihedrals','hbonds']):

        """Compute descriptors from trajectory:
        - Dihedral angles
        - CA distances
        - Hydrogen bonds

        Parameters
        ----------
        descriptors : string or list
            compute a single descriptor or a list ot them

        Raises
        ------
        KeyError
            Trajectory needs to be set beforehand.
        """
        if self.traj is None:
            raise KeyError("Trajectory not loaded. Call self.load_trajectory() first.")

        self.descriptors_ids = {} # used to store id of atoms associated with each feature

        descr_list = []

        for d in descriptors:
            if d == 'ca':
                descr_list.append(self._CA_DISTANCES())
            elif d == 'dihedrals':
                descr_list.append(self._ANGLES())
            elif d == 'hbonds':
                descr_list.append(self._HYDROGEN_BONDS())
            else:
                raise KeyError(f"descriptor: {d} not valid. Only 'ca','dihedrals','hbonds' are allowed.")
        
        self.descriptors = pd.concat(descr_list, axis=1)
        if self._DEV:
            print(f"Descriptors: {self.descriptors.shape}")
        assert len(self.colvar) == len(
            self.descriptors
        ), "mismatch between colvar and descriptor length."

    def identify_states(
        self,
        selected_cvs,
        bounds,
        logweights=None,
        fes_cutoff=5,
        sort_minima_by = 'cvs_grid',
        optimizer=None,
        use_jac = False,
        optimizer_kwargs=dict(),
        memory_saver=False, 
        splits=50
    ):
        """Label configurations based on free energy

        Parameters
        ----------
        selected_cvs : list of strings
            Names of the collective variables used for clustering
        bounds : tuple of list 
            Bounds for the cvs ([min,max]*n_cvs)
        logweights : pandas.DataFrame, np.array or string , optional
            Logweights used for FES calculation, by default None
        fes_cutoff : float, optional
            Cutoff used to select only low free-energy configurations, by default 5
        sort_minima_by : string, optional
            Sort labels based on `energy`, `cvs`, or `cvs_grid` values, by default `cvs_grid`.
        optimizer : optional
            Method for finding local minima, by default None
        optimizer_kwargs : optional
            Arguments for optimizer, by default dict()
        memory_saver : bool, optional
            Memory saver option for basin selection, by default False
        splits : int, optional
            Divide data in `splits` chuck, by default 50

        """

        # retrieve logweights
        if logweights is None:
            w = None
            if ".bias" in self.colvar.columns:
                print(
                    "WARNING: a field with .bias is present in colvar, but it is not used for the FES.",file=sys.stderr
                )
        else:
            if isinstance(logweights, str):
                if "*" in logweights:
                    w = self.colvar.filter(regex=logweights.replace('*','')).sum(axis=1).values / self.kbt
                else:
                    w = self.colvar[logweights].values / self.kbt
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

        # store selected cvs
        self.selected_cvs = selected_cvs

        # Compute fes
        self.approximate_FES(
            selected_cvs, 
            bw_method=None, 
            logweights=w
        )
        if use_jac:
            raise NotImplementedError("Not yet implemented")

        self.minima = local_minima(self.fes, bounds, method=optimizer, method_kwargs=optimizer_kwargs)
        
        # sort minima based on first CV
        if sort_minima_by == 'energy':
            pass #already sorted
        elif sort_minima_by == 'cvs' :
            # sort first by 1st cv, then 2nd, ...
            x = self.minima
            self.minima = x [ np.lexsort( np.round(np.flipud(x.T),2) ) ] 
        elif sort_minima_by == 'cvs_grid' :
            # sort based on a binning of the cvs (10 bins per each direction),
            # along 1st cv, then 2nd, ... 
            x = self.minima
            y = (x - [ bound[0] for bound in bounds ]) 
            y /= np.asarray( [ bound[1]-bound[0] for bound in bounds ]) / 10
            self.minima = x [ np.lexsort( np.round(np.flipud(y.T),0) ) ] 
        else:
            raise KeyError(f'Key {sort_minima_by} not allowed. Valid values: "energy","cvs","cvs_grid".')

        # Assign basins and select based on FES cutoff
        self.basins = self._basin_selection(
            self.minima,
            fes_cutoff=fes_cutoff,
            memory_saver=memory_saver, 
            splits=splits
        )
        
        self.n_basins = len(self.basins['basin'].unique())
        self.bounds = bounds

    def collect_data(self, only_selected_cvs=False):
        """Prepare dataframe with: CVs, labels and descriptors

        Parameters
        ----------
        only_selected_cvs : bool, optional
            save only selected CVs for labeling

        Returns
        -------
        pandas.DataFrame
            dataset with all data

        Raises
        ------
        KeyError
            Basins labels needs to be set beforehand.
        """

        if self.basins is None:
            raise KeyError("Basins not selected. Call identify_states() first.")

        return pd.concat(
            [
                self.colvar[self.selected_cvs] if only_selected_cvs else self.colvar,
                self.basins,
                self.descriptors,
            ],
            axis=1,
        )

    def approximate_FES(
        self, collective_vars, bw_method=None, logweights=None
    ):
        """Approximate Free Energy Surface (FES) in the space of collective_vars through Gaussian Kernel Density Estimation

        Args:
            collective_vars (numpy.ndarray or pd.Dataframe): List of sampled collective variables with dimensions [num_timesteps, num_CVs]
            bounds (list of tuples): (min, max) bounds for each collective Variable
            num (int, optional): [description]. Defaults to 100.
            bw_method ('scott', 'silverman' or a scalar, optional): Bandwidth method used in GaussianKDE. Defaults to None ('scotts' factor).
            logweights (arraylike log weights, optional): [description]. Defaults to None (uniform weights).

        Returns:
            [type]: [description]
        """
        empirical_centers = self.colvar[collective_vars].to_numpy()
        self.KDE = gaussian_kde(empirical_centers,bw_method=bw_method,logweights=logweights)
        self.fes = lambda X: -self.kbt*self.KDE.logpdf(X)   
        return self.fes

    def _basin_selection(
        self, minima, fes_cutoff=5, memory_saver=False, splits=50
    ):
    
        positions = self.KDE.dataset
        norms = np.linalg.norm((positions[:,np.newaxis,:] - minima), axis=2)
        classes = np.argmin(norms, axis=1)
        fes_at_minima = self.fes(minima)
        ref_fes = np.asarray([fes_at_minima[idx] for idx in classes])
        # Very slow
        if memory_saver:
            chunks = np.array_split(positions, splits, axis=0)
            fes_pts = []
            if self._DEV:
                for chunk in tqdm(chunks):
                    fes_pts.append(self.fes(chunk))
            else:
                for chunk in chunks:
                    fes_pts.append(self.fes(chunk))
            fes_pts = np.hstack(fes_pts)
        else:
            fes_pts = self.fes(positions)
        mask = (fes_pts - ref_fes) < fes_cutoff
        df = pd.DataFrame(data=classes, columns=["basin"])
        df["selection"] = mask
        return df

    # DESCRIPTORS COMPUTATION

    def _CA_DISTANCES(self):
        sel = self.traj.top.select("name CA")

        pairs = [(i, j) for i, j in itertools.combinations(sel, 2)]
        dist = md.compute_distances(self.traj, pairs)

        # Labels
        label = lambda i, j: "DIST. %s%s -- %s%s" % (
            self.traj.top.atom(i),
            "s" if self.traj.top.atom(i).is_sidechain else "",
            self.traj.top.atom(j),
            "s" if self.traj.top.atom(j).is_sidechain else "",
        )

        names = [label(i, j) for (i, j) in pairs]
        for (i,j) in pairs:
            self.descriptors_ids[label(i, j)] = [i,j]
        df = pd.DataFrame(data=dist, columns=names)
        return df

    def _HYDROGEN_BONDS(self):
        # H-BONDS DISTANCES / CONTACTS (donor-acceptor)
        # find donors (OH or NH)
        traj = self.traj
        _DEV = self._DEV
        donors = [
            at_i.index
            for at_i, at_j in traj.top.bonds
            if ((at_i.element.symbol == "O") | (at_i.element.symbol == "N"))
            & (at_j.element.symbol == "H")
        ]
        # keep unique
        donors = sorted(list(set(donors)))
        if _DEV:
            print("Donors:", donors)

        # find acceptors (O r N)
        acceptors = traj.top.select("symbol O or symbol N")
        if _DEV:
            print("Acceptors:", acceptors)

        # lambda func to avoid selecting interaction within the same residue
        atom_residue = lambda i: str(traj.top.atom(i)).split("-")[0]
        # compute pairs
        pairs = [
            (min(x, y), max(x, y))
            for x in donors
            for y in acceptors
            if (x != y) and (atom_residue(x) != atom_residue(y))
        ]
        # remove duplicates
        pairs = sorted(list(set(pairs)))

        # compute distances
        dist = md.compute_distances(traj, pairs)
        # labels
        label = lambda i, j: "HB_DIST %s%s -- %s%s" % (
            traj.top.atom(i),
            "s" if traj.top.atom(i).is_sidechain else "",
            traj.top.atom(j),
            "s" if traj.top.atom(j).is_sidechain else "",
        )

        # basename = 'hb_'
        # names = [ basename+str(x)+'-'+str(y) for x,y in  pairs]
        names = [label(x, y) for x, y in pairs]
        for (x,y) in pairs:
            self.descriptors_ids[label(x,y)] = [x,y]
            
        df_HB_DIST = pd.DataFrame(data=dist, columns=names)

        # compute contacts
        contacts = self.contact_function(dist, r0=0.35, d0=0, n=6, m=12)
        # labels
        # basename = 'hbc_'
        # names = [ basename+str(x)+'-'+str(y) for x,y in pairs]
        label = lambda i, j: "HB_CONTACT %s%s -- %s%s" % (
            traj.top.atom(i),
            "s" if traj.top.atom(i).is_sidechain else "",
            traj.top.atom(j),
            "s" if traj.top.atom(j).is_sidechain else "",
        )
        names = [label(x, y) for x, y in pairs]
        for (x,y) in pairs:
            self.descriptors_ids[label(x,y)] = [x,y]

        df = pd.DataFrame(data=contacts, columns=names)
        df = df.join(df_HB_DIST)
        return df

    def _ANGLES(self):
        # DIHEDRAL ANGLES
        # phi,psi --> backbone
        # chi1,chi2 --> sidechain

        values_list = []
        names_list = []

        for kind in ["phi", "psi", "chi1", "chi2"]:
            names, values = self._get_dihedrals(kind, sincos=True)
            names_list.extend(names)
            values_list.extend(values)

        df = pd.DataFrame(data=np.asarray(values_list).T, columns=names_list)
        return df

    def _get_dihedrals(self, kind="phi", sincos=True):
        traj = self.traj
        # retrieve topology
        table, _ = traj.top.to_dataframe()

        # prepare list for appending
        dihedrals = []
        names, values = [], []

        if kind == "phi":
            dihedrals = md.compute_phi(traj)
        elif kind == "psi":
            dihedrals = md.compute_psi(traj)
        elif kind == "chi1":
            dihedrals = md.compute_chi1(traj)
        elif kind == "chi2":
            dihedrals = md.compute_chi2(traj)
        else:
            raise KeyError("supported values: phi,psi,chi1,chi2")

        idx_list = dihedrals[0]
        for i, idx in enumerate(idx_list):
            # find residue id from topology table
            # res = table['resSeq'][idx[0]]
            # name = 'dih_'+kind+'-'+str(res)
            res = table["resName"][idx[0]] + table["resSeq"][idx[0]].astype("str")
            name = "BACKBONE " + kind + " " + res
            if "chi" in kind:
                name = "SIDECHAIN " + kind + " " + res
            names.append(name)
            values.append(dihedrals[1][:, i])
            self.descriptors_ids[name] = list(idx)
            
            if sincos:
                # names.append('cos_'+kind+'-'+str(res))
                name = "BACKBONE " + "cos_" + kind + " " + res
                if "chi" in kind:
                    name = "SIDECHAIN " + "cos_" + kind + " " + res
                names.append(name)
                values.append(np.cos(dihedrals[1][:, i]))
                self.descriptors_ids[name] = list(idx)

                # names.append('sin_'+kind+'-'+str(res))
                name = "BACKBONE " + "sin_" + kind + " " + res
                if "chi" in kind:
                    name = "SIDECHAIN " + "sin_" + kind + " " + res
                names.append(name)
                values.append(np.sin(dihedrals[1][:, i]))
                self.descriptors_ids[name] = list(idx)

        return names, values

    def contact_function(self, x, r0=1.0, d0=0, n=6, m=12):
        # (see formula for RATIONAL) https://www.plumed.org/doc-v2.6/user-doc/html/switchingfunction.html
        return (1 - np.power(((x - d0) / r0), n)) / (1 - np.power(((x - d0) / r0), m))

    def sample(self, n_configs, regex_filter = '.*', states_subset=None):
        features = self.descriptors.filter(regex=regex_filter).columns.values
        config_list = []
        labels = []
        if states_subset is None:
            states_subset = range(self.n_basins)
        for basin in states_subset:
            #select basin
            df = self.descriptors.loc[ (self.basins['basin'] == basin) & (self.basins['selection'] == True)]
            #select descriptors and sample
            config_i = df.filter(regex=regex_filter).sample(n=n_configs).values
            config_list.append(config_i)
            labels.extend([basin]*n_configs)
        labels = np.array(labels, dtype=np.int_)
        configurations = np.vstack(config_list)
        return Sample(configurations, features, labels, scale=True)

class Sample:
    def __init__(self, configurations, features, labels, scale=False):
        self.unscaled_configurations = configurations
        self.configurations = configurations
        self.features = features
        self.labels = labels
        if scale:
            self.scale()
    def scale(self):
        self.scaler = StandardScaler(with_mean=True)
        self.scaler.fit(self.configurations)
        self.configurations = self.scaler.transform(self.unscaled_configurations)
    def train_test_dataset(self, **kwargs):
        return train_test_split(self.configurations, self.labels, **kwargs)
    @property
    def dataset(self):
        return [self.configurations, self.labels]