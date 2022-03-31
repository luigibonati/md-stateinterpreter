"""
Unit and regression test for the MD module.
"""

# Import package, test suite, and other packages as needed
import pytest
from stateinterpreter import Classifier, identify_metastable_states, load_dataframe, load_trajectory, compute_descriptors, prepare_training_dataset

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

    traj = load_trajectory(traj_dict,stride=stride)
    descriptors, feats_info = compute_descriptors(traj)

    descriptors_loaded = load_dataframe(descr_file, stride = stride)
    assert descriptors.shape == descriptors_loaded.shape
    
    cvs_list = ['deep.node-4','deep.node-3','deep.node-2']
    selected_cvs = cvs_list[:n_cvs]
    print('cvs: '+" - ".join(selected_cvs))
    optimizer_kwargs = {
        'sampling': sampling_scheme
    }
    kBT = 2.8
    states_labels = identify_metastable_states(colvar, selected_cvs, kBT, bandwidth=0.1, optimizer_kwargs = optimizer_kwargs, sort_minima_by=sort_minima_by)

    dataset, features = prepare_training_dataset(descriptors,states_labels,10)

    classifier = Classifier(dataset,features)
    classifier.compute(0.1)

if __name__ == "__main__":
    test_chignolin_pipeline()