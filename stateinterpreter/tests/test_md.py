"""
Unit and regression test for the MD module.
"""

# Import package, test suite, and other packages as needed
import pytest
from stateinterpreter import Classifier, identify_metastable_states, load_dataframe, descriptors_from_traj, sample

@pytest.mark.parametrize("n_cvs", [1,2])
@pytest.mark.parametrize("sort_minima_by", ['energy','cvs','cvs_grid'])
@pytest.mark.parametrize("sampling_scheme", ['data_driven','uniform'])
def test_chignolin_pipeline(sort_minima_by,n_cvs, sampling_scheme):
    """Identify metastable states based on FES, clustering with differen no. of CVs"""
    stride = 20
    folder = 'stateinterpreter/data/test-chignolin/'
    colvar_file = folder + 'COLVAR'
    
    colvar = load_dataframe(colvar_file, stride=stride)

    descr_file = folder + "DESCRIPTORS.csv"

    traj_dict = {
        'trajectory' : folder+'traj.dcd',
        'topology' : folder+'topology.pdb'
    }

    descriptors, _ = descriptors_from_traj(traj_dict, stride= stride)
    descriptors_loaded = load_dataframe(descr_file, stride = stride)
    assert descriptors.shape == descriptors_loaded.shape
    
    cvs_list = ['deep.node-4','deep.node-3','deep.node-2']
    selected_cvs = cvs_list[:n_cvs]
    print('cvs: '+" - ".join(selected_cvs))
    optimizer_kwargs = {
        'sampling': sampling_scheme
    }
    kBT = 2.8
    states_labels = identify_metastable_states(colvar, selected_cvs, kBT, optimizer_kwargs = optimizer_kwargs, sort_minima_by=sort_minima_by)

if __name__ == "__main__":
    test_chignolin_pipeline()