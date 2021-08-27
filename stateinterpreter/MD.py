import pandas as pd
import mdtraj as md
import numpy as np
import scipy.stats as st

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
        self.FES = (sampled_positions, f)
        return self.FES