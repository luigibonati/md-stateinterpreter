# imports
from typing import Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
import itertools

from tqdm import tqdm
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.optimize import minimize

from .numerical_utils import gaussian_kde

"""
OUTLINE
=======
1a. Load collective variables
   - from FILE or pd.DataFrame
1b. Load descriptors
   - from FILE or pd.DataFrame
2. (optional: load trajectory and compute descriptors)
3. Compute FES --> assign states to configs
4. Dataframe(CVs, descriptors, labels)
"""

"""
TODO
====
- weights or logweights
- basins or states?
- name functions
- mismatch approximate_fes (pietro)
"""

def is_plumed_file(filename):
    """
    Check if given file is in PLUMED format.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    bool
        wheter is a plumed output file
    """   
    headers = pd.read_csv( filename, sep=" ", skipinitialspace=True, nrows=0 )
    is_plumed = True if " ".join(headers.columns[:2]) == "#! FIELDS" else False
    return is_plumed

def plumed_to_pandas(filename="./COLVAR"):
    """
    Load a PLUMED file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    # Read header 
    headers = pd.read_csv( filename, sep=" ", skipinitialspace=True, nrows=0 )
    # Discard #! FIELDS
    headers = headers.columns[2:]
    # Load dataframe and use headers for columns names
    df = pd.read_csv(
        filename,
        sep=' ',
        skipinitialspace=True,
        header=None,
        skiprows=range(skip_rows),
        names=headers,
        comment="#",
    )

    return df

def load_dataframe(data, **kwargs):
    """Load dataframe from object or from file.

    Parameters
    ----------
    data : str or pandas.DataFrame
        input data

    Returns
    -------
    pandas.DataFrame
        Dataframe

    Raises
    ------
    TypeError
        if data is not a valid type
    """
    # check if data is Dataframe
    if type(data) == pd.DataFrame:
        df = data
    # or is a string
    elif type(data) == str:
        filename = data
        # check if file is in PLUMED format
        if is_plumed_file(filename):
            df = plumed_to_pandas(filename)
        # else use read_csv with optional kwargs
        else:
            df = pd.read_csv(filename, **kwargs)
    else:
        raise TypeError(f"{data}: Accepted types are \'pandas.Dataframe\' or \'str\'")          

    return df

class Loader:

    def __init__(self, colvar = None, descriptors = None, kbt = 2.5, stride=1, _DEV=False, **kwargs):
        """Prepare inputs for stateinterpreter

        Parameters
        ----------
        colvar : pandas.DataFrame or string
            collective variables
        descriptors : pandas.DataFrame or string, optional
            input features, by default None
        kbt : float, optional
            temperature [KbT], by default 2.5
        stride : int, optional
            keep data every stride, by default 1
        _DEV : bool, optional
            enable dev. mode, by default False
        """
        # collective variables data
        self.colvar = load_dataframe(colvar, **kwargs)      
        self.colvar = self.colvar.iloc[::stride, :]
        if _DEV:
            print(f'Collective variables: {self.colvar.values.shape}')

        # descriptors data
        if descriptors is not None:
            self.descriptors = load_dataframe(descriptors, **kwargs)
            self.descriptors = self.descriptors.iloc[::stride, :]
            if 'time' in self.descriptors.columns:
                self.descriptors = self.descriptors.drop('time',axis='columns')
            if _DEV:
                print(f'Descriptors: {self.descriptors.shape}')
            assert len(self.colvar) == len(self.descriptors), "mismatch between colvar and descriptor length."
        
        # save attributes
        self.kbt = kbt
        self.stride = stride
        self._DEV = _DEV

        #initialize attributes to None
        self.traj = None
        self.basins = None

    def load_trajectory(self, traj_dict):
        """"Load trajectory with mdtraj.

        Parameters
        ----------
        traj_dict : dict
            dictionary containing trajectory and topology (optional) file
        """
        traj_file = traj_dict['trajectory']
        topo_file = traj_dict['topology'] if 'topology' in traj_dict else None

        self.traj = md.load(traj_file, top=topo_file, stride=self.stride)
        
        assert len(self.traj) == len(self.colvar), f"length traj ({len(self.traj)}) != length colvar ({len(self.colvar)})"

    def compute_descriptors(self):
        """ Compute descriptors from trajectory:
        - Dihedral angles
        - CA distances
        - Hydrogen bonds

        Raises
        ------
        KeyError
            Trajectory needs to be set beforehand.
        """
        if self.traj == None:
            raise KeyError('Trajectory not loaded. Call self.load_trajectory() first.')

        ca = self._CA_DISTANCES()
        hb = self._HYDROGEN_BONDS()
        ang = self._ANGLES()

        self.descriptors = pd.concat([ca,hb,ang], axis=1)
        if self._DEV:
            print(f'Descriptors: {self.descriptors.shape}')
        assert len(self.colvar) == len(self.descriptors), "mismatch between colvar and descriptor length."

    def identify_states(self, selected_cvs, bounds, weights=None, num=100, fes_cutoff=5, memory_saver=False, splits=50):
        # retrieve weights
        if weights is None:
            if '.bias' in self.colvar.columns:
                print('WARNING: a field with .bias is present in colvar, but it is not used for the FES.')
        else:
            if isinstance(weights,str):
                w = self.colvar[weights].values
            elif isinstance(weights,pd.DataFrame):
                w = weights.values
            elif isinstance(weights,np.ndarray):
                w = weights
            else:
                raise TypeError(f"{weights}: Accepted types are \'pandas.Dataframe\', \'str\' or \'numpy.ndarray\' ")
            if w.ndim != 1:
                raise ValueError(f"{weights}: 1D array is required for weights")
        
        # store selected cvs
        self.selected_cvs = selected_cvs
        # compute fes
        self.approximate_FES(selected_cvs, bounds, num=num, bw_method=None, weights=weights) 
        # assign basins and select based on FES cutoff
        self.basins = self._basin_selection(fes_cutoff=fes_cutoff, memory_saver=memory_saver,splits=splits)

    def collect_data(self, only_selected_cvs = False):
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
            raise KeyError('Basins not selected. Call identify_states() first.')

        return pd.concat( [
            self.colvar[self.selected_cvs] if only_selected_cvs else self.colvar, 
            self.basins, 
            self.descriptors 
            ], axis=1)

    # DEPRECATED, TO REMOVE
    def load(self, collective_vars, bounds, num=100, fes_cutoff=5, memory_saver=False, splits=50,bw_method=None,weights=None):
        self.approximate_FES(collective_vars, bounds, num=num, bw_method=None, weights=None)
        CVs = self.colvar[collective_vars]
        basins = self._basin_selection(fes_cutoff=fes_cutoff, memory_saver=memory_saver,splits=splits)
        CA_DIST = self._CA_DISTANCES()
        HB = self._HYDROGEN_BONDS()
        ANGLES = self._ANGLES()
        return pd.concat([CVs, basins, CA_DIST,HB,ANGLES], axis=1)

    def approximate_FES(self, collective_vars, bounds, num=100, bw_method=None, weights=None):
        ndims = len(collective_vars)
        positions = np.array(self.colvar[collective_vars])
        _FES = gaussian_kde(positions,bw_method=bw_method,weights=weights )
        self._FES_KDE = _FES
        _1d_samples = [np.linspace(vmin, vmax, num) for (vmin, vmax) in bounds]
        meshgrids = np.meshgrid(*_1d_samples)
        sampled_positions = np.array([np.ravel(coord) for coord in meshgrids])
        f = np.reshape(_FES.logpdf(sampled_positions), (num,)*ndims)
        f *= -self.kbt
        f -= np.min(f)

        self.FES = (meshgrids, f)
        return self.FES

    def plot_FES(self, bounds= None, names = ['Variable 1', 'Variable 2']):
        try:
            self.FES
        except NameError:
            print("Free energy surface hasn't been computed. Use approximate_FES function.")
        else:
            pass
        sampled_positions, f = self.FES
        FES_dims = f.ndim
        if FES_dims == 1:
            fig,ax = plt.subplots(dpi=100)
            xx = sampled_positions[0]
            ax.plot(xx, f)
            ax.set_xlabel(names[0])
            ax.set_ylabel('FES [kJ/mol]')
            return (fig, ax)
        elif FES_dims == 2:          
            xx = sampled_positions[0]
            yy = sampled_positions[1]

            if not bounds:
                levels = np.linspace(1,30,10)
            else:
                levels = np.linspace(bounds[0], bounds[1], 10)

            fig,ax = plt.subplots(dpi=100)
            cfset = ax.contourf(xx, yy, f, levels=levels, cmap='Blues')
            # Contour plot
            cset = ax.contour(xx, yy, f, levels=levels, colors='k')
            # Label plot
            ax.clabel(cset, inline=1, fontsize=10)

            cbar = plt.colorbar(cfset)
            
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
            cbar.set_label('FES [kJ/mol]')
            return (fig, ax)
        else:
            raise ValueError("Maximum number of dimensions over which to plot is 2")
    
    def find_minima(self):
        # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        Takes an array and detects the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """
        _DEV= self._DEV
        sampled_positions, f = self.FES
        # define an connected neighborhood http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
        neighborhood = generate_binary_structure(f.ndim,2)
        # apply the local minimum filter; all locations of minimum value in their neighborhood are set to 1 http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
        local_min = (minimum_filter(f, footprint=neighborhood)==f)
        # local_min is a mask that contains the peaks we are looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask. we create the mask of the background
        background = (f==0)
        # 
        # a little technicality: we must erode the background in order to successfully subtract it from local_min, otherwise a line will appear along the background border (artifact of the local minimum filter) http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        # 
        # we obtain the final mask, containing only peaks, by removing the background from the local_min mask
        detected_minima = local_min ^ eroded_background
        detected_minima = np.where(detected_minima)
        presumed_minima = []
        
        for idxs in zip(*detected_minima):
            minima =[]
            for coord in sampled_positions:
                minima.append(coord[idxs])
            presumed_minima.append(np.array(minima))
        presumed_minima = np.array(presumed_minima)
        if _DEV:
            print(f"Detected {len(presumed_minima)} minima")
        obj_fun = lambda x: -self.kbt*self._FES_KDE.logpdf(x)
        
        real_minima = []
        for minima in presumed_minima:
            
            res = minimize(obj_fun, minima)
            if res.success:
                real_minima.append(res.x)
            else:
                real_minima.append(minima)
                if _DEV:
                    print(f"Unable to converge minima at {minima}, using approximated one")
        real_minima = np.array(real_minima)
        
        if _DEV:
            print("True\t Approx\t Delta")
            for i in range(len(real_minima)):
                print(f"{np.around(real_minima[i], decimals=2)}\t {np.around(presumed_minima[i], decimals=2)}\t {np.abs(np.around(real_minima[i] - presumed_minima[i], decimals=2))}\t")
        #Sorting w.r.t. first dim, to be stabilized 
        argsortmin = np.argsort(real_minima, axis = 0)
        return real_minima[argsortmin[:,0]]
    
    def _basin_selection(self, fes_cutoff=5, minima=None, memory_saver=False, splits=50):
        if not minima:
            minima = self.find_minima()
        positions = self._FES_KDE.dataset.T
        norms = np.linalg.norm((positions[...,np.newaxis] - minima.T),axis=1)
        classes = np.argmin(norms, axis = 1)
        fes_at_minima = -self.kbt*self._FES_KDE.logpdf(minima.T)
        ref_fes = np.asarray([fes_at_minima[idx] for idx in classes])
        #Very slow
        if memory_saver:
            chunks = np.array_split(positions.T, splits, axis = 1)
            fes_pts = []
            if self._DEV:
                for chunk in tqdm(chunks):
                    fes_pts.append(-self.kbt*self._FES_KDE.logpdf(chunk))
            else:
                for chunk in chunks:
                    fes_pts.append(-self.kbt*self._FES_KDE.logpdf(chunk))
            fes_pts = np.hstack(fes_pts)
        else:
            fes_pts = -self.kbt*self._FES_KDE.logpdf(positions.T)
        mask = (fes_pts - ref_fes) < fes_cutoff
        df = pd.DataFrame(data=classes, columns=['basin'])
        df['selection'] = mask
        return df
        
    def _CA_DISTANCES(self):
        sel = self.traj.top.select('name CA')

        pairs = [ (i,j) for i,j in itertools.combinations(sel,2) ]
        dist = md.compute_distances(self.traj,pairs)

        #Labels            
        label = lambda i,j : 'DIST. %s%s -- %s%s' % (
            self.traj.top.atom(i), 
            's' if self.traj.top.atom(i).is_sidechain else '',
            self.traj.top.atom(j), 
            's' if self.traj.top.atom(j).is_sidechain else ''
            )

        names = [label(i,j) for (i,j) in pairs]
        df = pd.DataFrame(data=dist,columns=names)
        return df

    def _HYDROGEN_BONDS(self):
        # H-BONDS DISTANCES / CONTACTS (donor-acceptor)
        # find donors (OH or NH)
        traj = self.traj
        _DEV= self._DEV
        donors = [ at_i.index for at_i,at_j in traj.top.bonds  
                    if ( ( at_i.element.symbol == 'O' ) | (at_i.element.symbol == 'N')  ) & ( at_j.element.symbol == 'H')]
        # keep unique 
        donors = sorted( list(set(donors)) )
        if _DEV:
            print('Donors:',donors)

        # find acceptors (O r N)
        acceptors = traj.top.select('symbol O or symbol N')
        if _DEV:
            print('Acceptors:',acceptors)

        # lambda func to avoid selecting interaction within the same residue
        atom_residue = lambda i : str(traj.top.atom(i)).split('-')[0] 
        # compute pairs
        pairs = [ (min(x,y),max(x,y)) for x in donors for y in acceptors if (x != y) and (atom_residue(x) != atom_residue(y) ) ]
        # remove duplicates
        pairs = sorted(list(set(pairs)))

        # compute distances
        dist = md.compute_distances(traj,pairs)
        # labels
        label = lambda i,j : 'HB_DIST %s%s -- %s%s' % (
            traj.top.atom(i), 
            's' if traj.top.atom(i).is_sidechain else '',
            traj.top.atom(j), 
            's' if traj.top.atom(j).is_sidechain else ''
            )
        
        #basename = 'hb_'
        #names = [ basename+str(x)+'-'+str(y) for x,y in  pairs]
        names = [ label(x,y) for x,y in pairs]

        df_HB_DIST = pd.DataFrame(data=dist,columns=names)

        # compute contacts
        contacts = self.contact_function(dist,r0=0.35,d0=0,n=6,m=12)
        # labels
        #basename = 'hbc_'
        #names = [ basename+str(x)+'-'+str(y) for x,y in pairs]
        label = lambda i,j : 'HB_CONTACT %s%s -- %s%s' % (
            traj.top.atom(i), 
            's' if traj.top.atom(i).is_sidechain else '',
            traj.top.atom(j), 
            's' if traj.top.atom(j).is_sidechain else ''
            )
        names = [ label(x,y) for x,y in pairs]
        df = pd.DataFrame(data=contacts,columns=names)
        df = df.join(df_HB_DIST)
        return df

    def _ANGLES(self):
        # DIHEDRAL ANGLES
        # phi,psi --> backbone
        # chi1,chi2 --> sidechain

        values_list = []
        names_list = []

        for kind in ['phi','psi','chi1','chi2']:
            names, values = self._get_dihedrals(kind,sincos=True)
            names_list.extend(names)
            values_list.extend(values)

        df = pd.DataFrame(data=np.asarray(values_list).T,columns=names_list)
        return df

    def _get_dihedrals(self,kind='phi',sincos=True):
        traj = self.traj
        #retrieve topology
        table, _ = traj.top.to_dataframe()

        #prepare list for appending
        dihedrals = []
        names,values = [],[]
        
        if kind == 'phi':
            dihedrals = md.compute_phi(traj)
        elif kind == 'psi':
            dihedrals = md.compute_psi(traj)
        elif kind == 'chi1':
            dihedrals = md.compute_chi1(traj)
        elif kind == 'chi2':
            dihedrals = md.compute_chi2(traj)
        else:
            raise KeyError('supported values: phi,psi,chi1,chi2')

        idx_list = dihedrals[0]
        for i, idx in enumerate(idx_list):
            #find residue id from topology table
            #res = table['resSeq'][idx[0]]
            #name = 'dih_'+kind+'-'+str(res)
            res = table['resName'][idx[0]]+table['resSeq'][idx[0]].astype('str')
            name = 'BACKBONE '+ kind + ' ' + res
            if 'chi' in kind:
                name = 'SIDECHAIN '+ kind + ' ' + res
            names.append(name)
            values.append(dihedrals[1][:,i])
            if sincos:
                #names.append('cos_'+kind+'-'+str(res)) 
                name = 'BACKBONE '+ 'cos_'+ kind + ' ' + res
                if 'chi' in kind:
                    name = 'SIDECHAIN '+ 'cos_'+ kind + ' ' + res
                names.append(name)
                values.append(np.cos( dihedrals[1][:,i] ))

                #names.append('sin_'+kind+'-'+str(res)) 
                name = 'BACKBONE '+ 'sin_'+ kind + ' ' + res
                if 'chi' in kind:
                    name = 'SIDECHAIN '+ 'sin_'+ kind + ' ' + res
                names.append(name)
                values.append(np.sin( dihedrals[1][:,i] ))
        return names, values

    def contact_function(self, x,r0=1.,d0=0,n=6,m=12):
        # (see formula for RATIONAL) https://www.plumed.org/doc-v2.6/user-doc/html/switchingfunction.html
        return ( 1-np.power(((x-d0)/r0),n) ) / ( 1-np.power(((x-d0)/r0),m) )



        


        