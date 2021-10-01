"""
Unit and regression test for the MD module.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import stateinterpreter
import pandas as pd

from stateinterpreter.MD import Loader
from stateinterpreter.io import load_dataframe

def test_load_colvar():
    """Test loader initialization w/only collective variables"""

    folder = 'stateinterpreter/data/test-chignolin/'
    colvar_file = folder + 'COLVAR'

    #load COLVAR from file
    data = Loader(colvar_file, kbt=2.8, stride=20, _DEV=True)

    #set COLVAR from dataframe
    colvar_df = load_dataframe(colvar_file)
    data2 = Loader(colvar_df, kbt=2.8, stride=20, _DEV=True)

def test_compute_descriptors():
    """Load compute descriptors"""
    
    folder = 'stateinterpreter/data/test-chignolin/'
    colvar_file = folder + 'COLVAR'
    traj_dict = {
        'trajectory' : folder+'traj.dcd',
        'topology' : folder+'topology.pdb'
    }

    # Compute descriptors from traj
    data = Loader(colvar_file, kbt=2.8, stride=20, _DEV=True)
    data.load_trajectory(traj_dict)
    data.compute_descriptors()

    # Load descriptors from file
    descr_file = folder+"DESCRIPTORS.csv"
    #data.descriptors.to_csv(descr_file,index=False) #to create example, set stride to 1 above
    data2 = Loader(colvar_file, descr_file, kbt=2.8, stride=20, _DEV=True)

    assert data.descriptors.shape == data2.descriptors.shape

@pytest.mark.skip
@pytest.mark.parametrize("n_cvs", [1,2])
def test_identify_states(n_cvs):
    """Identify metastable states based on FES, clustering with differen no. of CVs"""

    folder = 'stateinterpreter/data/test-chignolin/'
    colvar_file = folder + 'COLVAR'
    descr_file = folder + "DESCRIPTORS.csv"

    cvs_list = ['deep.node-4','deep.node-3','deep.node-2']
    selected_cvs = cvs_list[:n_cvs]
    print('cvs: '+" - ".join(selected_cvs))

    bounds = [(-1.,1.)]*n_cvs

    data = Loader(colvar_file, descr_file, kbt=2.8, stride=1, _DEV=True)
    data.identify_states(selected_cvs,bounds,num=50)
    df = data.collect_data()
    print(df.shape)

if __name__ == "__main__":
    print('>>>>> test_load_colvar')
    test_load_colvar()
    print('>>>>> test_compute_descriptors')
    test_compute_descriptors()
    print('>>>>> test_identify_states')
    test_identify_states(2)