import pandas as pd
import mdtraj as md
import numpy as np

class Loader():
    def __init__(data_path, file_dict, stride=10):
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