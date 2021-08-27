import pandas as pd
import mdtraj as md
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

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
            cset = ax.contour(xx, yy, f, levels=levels, colors='k', lw=0.77)
            # Label plot
            ax.clabel(cset, inline=1, fontsize=10)

            cbar = plt.colorbar(cfset)
            
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
            cbar.set_label('FES [kJ/mol]')
            return (fig, ax)
        else:
            raise ValueError("Maximum number of dimensions over which to plot is 2")
