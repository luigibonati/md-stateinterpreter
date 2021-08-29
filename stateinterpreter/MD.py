import re
from numpy.lib.type_check import real
import pandas as pd
import mdtraj as md
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.optimize import minimize

class Loader:
    def __init__(self, data_path, file_dict, stride=10):
        '''
            file_dict = {'trajectory': '**.dcd', 'topology': '**.pdb', 'collective_vars': 'COLVAR' }
        '''
        self._base_path = data_path
        self.kbt = 2.827
        traj_file = data_path + file_dict['trajectory']
        topo_file = data_path + file_dict['topology']
        colvar_file = data_path + file_dict['collective_vars']
        self.traj = md.load(traj_file, top=topo_file,stride=stride)
        headers = pd.read_csv(colvar_file,sep=' ',skipinitialspace=True, nrows=0).columns[2:]
        self.colvar = pd.read_csv(colvar_file,sep=' ',skipinitialspace=True, header=None,skiprows=1,names=headers,comment='#')  
        self.colvar = self.colvar.iloc[::stride, :]
        self.colvar.index = np.arange(len(self.colvar))
        assert len(self.traj) == len(self.colvar)
        
    def approximate_FES(self, collective_vars, bounds, num=100):
        ndims = len(collective_vars)
        positions = np.array(self.colvar[collective_vars]).T
        _FES = st.gaussian_kde(positions)
        self._FES_KDE = _FES
        _1d_samples = [np.linspace(vmin, vmax, num) for (vmin, vmax) in bounds]
        meshgrids = np.meshgrid(*_1d_samples)
        sampled_positions = np.array([np.ravel(coord) for coord in meshgrids])
        f = np.reshape(_FES.logpdf(sampled_positions).T, (num,)*ndims)
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
    def find_minima(self, _DEV=False):
        # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        Takes an array and detects the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """
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
        for i,j in zip(*detected_minima):
            minima =[]
            for coord in sampled_positions:
                minima.append(coord[i,j])
            presumed_minima.append(np.array(minima))
        presumed_minima = np.array(presumed_minima)
        if _DEV:
            print(f"Detected {len(presumed_minima)} minima, approximately at")
            for minima in presumed_minima:
                print(minima)
        obj_fun = lambda x: -self.kbt*self._FES_KDE.logpdf(x)
        
        real_minima = []
        for minima in presumed_minima:
            
            res = minimize(obj_fun, minima)
            if res.success:
                real_minima.append(res)
            else:
                real_minima.append(minima)
                if _DEV:
                    print(f"Unable to converge minima at {minima}, using approximated one")
        real_minima = np.array(real_minima)
        
        if _DEV:
            print("True\t Approx\t Delta")
            for i in range(len(real_minima)):
                print(f"{np.around(real_minima[i], decimals=2)}\t {np.around(presumed_minima[i], decimals=2)}\t {np.abs(np.around(real_minima[i] - presumed_minima, decimals=2))}\t")
        return real_minima

        