import nglview
import matplotlib
import matplotlib.cm as cm
import numpy as np
from scipy.sparse import csr_matrix
from .._configs import *
from .plot import paletteFessa
from time import sleep


def visualize_features(trajectory, states_labels, classes_names, relevant_features, feats_info, state = 0, n_feat_per_state=3, representation = 'licorice'):
    """Visualize snapshots of each state highlighting the relevant features for a given state. 

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        MD trajectory
    states_labels : pd.DataFrame
        labels
    classes_names : list
        names of the classes
    relevant_features : dict
        features selected by Lasso
    feats_info : pd.DataFrame
        descriptors information (atoms involved)
    state : int, optional
        state for which the features are displayed, by default 0
    n_feat_per_state : int, optional
        number of features to be highlighted, by default 3
    representation : str, optional
        type of representation (licorice,cartoon,ball-and-stick), by default 'licorice'

    Returns
    -------
    nglview viewer
        View object
    """
    # sample one frame per state
    frames = [states_labels [( states_labels['labels'] == i ) & ( states_labels['selection'] ) ].sample(1).index.values[0] for i in classes_names.keys() ]
    traj = trajectory[frames]
    traj.superpose(traj[0])

    # find atom ids of relevant features
    atom_ids = []

    features = relevant_features[ classes_names[state] ]
    for i, feature in enumerate(features):
        if i < n_feat_per_state:
            name = feature[2]
            atom_ids.append( feats_info[name]['atoms'] )

    # set up visualization
    view = nglview.show_mdtraj(traj, default=False)

    # representation 
    if representation == 'licorice':
        view.add_licorice('(not hydrogen)',opacity=0.35)
        view.add_licorice('(backbone) and (not hydrogen)',opacity=0.85)
    elif representation == 'cartoon':
        view.add_cartoon('protein',opacity=0.85)
    elif representation == 'ball-and-stick':
        view.add_ball_and_stick('(not hydrogen)',opacity=0.15)
        view.add_ball_and_stick('(backbone) and (not hydrogen)',opacity=0.5)

    # colors
    colors = iter(['orange','green', 'purple', 'yellow', 'red'])

    # loop over relevant features
    for ids in atom_ids:
        ids_string = [str(p) for p in ids]
        selection = '@'+','.join(ids_string)

        color = next(colors)
        if len(ids) == 2: # distance
            #color = 'orange'
            atom_pair = [ '@'+p for p in ids_string ]
            view.add_distance(atom_pair=[atom_pair], color=color, label_visible=False)
            view.add_ball_and_stick(selection,color=color,opacity=0.75)
        elif len(ids) == 4: # angle
            #color = 'green'
            view.add_ball_and_stick(selection,color=color,opacity=0.75)

    return view

def compute_residue_score(classifier,reg,feats_info,n_residues):
    """Compute a residue score by aggregating all the features relevances by residues.

    Parameters
    ----------
    classifier : Classifier
        classifier object
    reg : float
        regularization magnitude
    feats_info : DataFrame
        descriptors information
    n_residues : int
        number of residues

    Returns
    -------
    dictionary
        residue score per each state
    """

    reg_idx = classifier._closest_reg_idx(reg)
    coefficients = classifier._coeffs[reg_idx]
    _classes = classifier._classes_labels[reg_idx]
    residue_score = dict()
    for idx, coef in enumerate(coefficients):
        score = np.zeros(n_residues)
        state_name = classifier.classes[_classes[idx]]
        coef = coef**2
        nrm = np.sum(coef)
        coef = coef/nrm
        if nrm < __EPS__:
            pass
        else:
            indices = csr_matrix(coef/nrm).indices
            for i in indices:
                if classifier._quadratic_kernel:
                    raise NotImplementedError("Residue Score not implemented for quadratic features")
                else:
                    feature_name = classifier.features[i]
                resnames = feats_info[feature_name]['group'].split('_')
                for res in resnames:
                    res_idx = int ( ''.join([n for n in res if n.isdigit()]) ) - 1
                    score[res_idx] += coef[i]/len(resnames)  
        residue_score[state_name] = score 
    return residue_score

def visualize_residue_score(trajectory, states_labels, classes_names, residue_score, representation = 'licorice', palette = 'Reds', state_frames=None, relevant_features = None,features_info=None):
    """Visualize snapshots of each state coloring the residues with the score per each state.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        MD trajectory
    states_labels : pd.DataFrame
        labels
    classes_names : list
        names of the classes
    residue_score : dict
        dictionary with the scores per each state
    representation : str, optional
        type of representation (licorice,cartoon,ball-and-stick), by default 'licorice'
    palette : str, optional
        color scheme, by default 'Reds'

    Returns
    -------
    nglview viewer
        View object
    """
# sample one frame per state
    if state_frames is None:
        frames = [states_labels [( states_labels['labels'] == i ) & ( states_labels['selection'] ) ].sample(1).index.values[0] for i in classes_names.values() ]
        print('frames:', frames)
    else:
        frames = state_frames
        
    traj = trajectory[frames]
    traj.superpose(traj[0])

    view = nglview.show_mdtraj(traj, default=False)

    if representation == 'licorice' :
        view.add_licorice('(not hydrogen)')
    elif representation == 'cartoon' :
        view.add_cartoon('protein')
    elif representation == 'ball_and_stick' :
        view.add_ball_and_stick('(not hydrogen)')

    for score in residue_score.values():
        for res,rescore in enumerate(score):
            if rescore>0:
                resnum = str(res+1)
                resnum.zfill(3)
                view.add_ball_and_stick(f'{resnum} and (not hydrogen) and',opacity=0.75)
                view.add_ball_and_stick(f'{resnum}',opacity=0.5)

    # highlight selected features
    if relevant_features is not None:
        atom_ids = []
        #features = relevant_features[ classes_names[state] ]
        features = relevant_features[ next(iter(relevant_features)) ]
        for i, feature in enumerate(features):
            name = feature[2]
            print(name,features_info[name]['atoms'])
            atom_ids.append( features_info[name]['atoms'] )

        #colors = iter(['orange','green', 'purple', 'yellow', 'red'])
        # loop over relevant features
        for ids in atom_ids:
            ids_string = [str(p) for p in ids]
            selection = '@'+','.join(ids_string)

            color = paletteFessa[1] #3#6 #'orange' #next(colors)
            if len(ids) == 2: # distance
                #color = 'orange'
                atom_pair = [ '@'+p for p in ids_string ]
                view.add_distance(atom_pair=[atom_pair], color=color, label_visible=False)
                #view.add_ball_and_stick(selection,color=color,opacity=0.75)
            elif len(ids) == 4: # angle
                #color = 'green'
                view.add_ball_and_stick(selection,color=color,opacity=0.75)

    # get color palette
    cmap = matplotlib.cm.get_cmap(palette, 11)
    palette = [matplotlib.colors.rgb2hex( cmap(i) ) for i in range(cmap.N)]

    # transform score in colors
    residue_colors = {}
    for i, state in enumerate( classes_names.values() ):
        colors = []
        for score in residue_score[ state ]:
            col = int(score*5*10)
            col = -1 if col > cmap.N-1 else col
            #col = 0
            colors.append( palette[col] ) 
        residue_colors[i] = colors
    
    # define observer function to allow changing colors with frame
    def on_change(change):
        frame = change.new
        frame_color = residue_colors[frame]
        frame_color = [c.replace('#', '0x') for c in frame_color]
        
        view._set_color_by_residue(view,frame_color)
        #view.update_licorice()
        sleep(0.1) # wait for the color update

    # convert to int
   
    # initialize set color by residue
    def _set_color_by_residue(self, colors, component_index=0, repr_index=0):
            self._remote_call('setColorByResidue',
                            target='Widget',
                            args=[colors, component_index, repr_index])

    if not hasattr(view, '_set_color_by_residue'):
        view._set_color_by_residue = _set_color_by_residue

    # set colors from state 0 
    frame_color = residue_colors[0]
    frame_color = [c.replace('#', '0x') for c in frame_color]
        
    view._set_color_by_residue(view,frame_color)
    view.observe(on_change, names=['frame'])

    return view

def visualize_protein_features(trajectory, states_labels, classes_names, residue_score, representation = 'licorice', state_frames=None, relevant_features = None, features_info=None, all_atoms=False, color=None):
    """Visualize snapshots of each state with the relevant features highlighted.
    """
    # sample one frame per state
    if state_frames is None:
        frames = [states_labels [( states_labels['labels'] == i ) & ( states_labels['selection'] ) ].sample(1).index.values[0] for i in classes_names.values() ]
        print('frames:', frames)
    else:
        frames = state_frames
        
    traj = trajectory[frames]
    traj.superpose(traj[0])

    view = nglview.show_mdtraj(traj, default=False)

    if representation == 'licorice' :
        view.add_licorice('(not hydrogen)')
    elif representation == 'cartoon' :
        view.add_cartoon('protein')
    elif representation == 'ball_and_stick' :
        view.add_ball_and_stick('(not hydrogen) and (backbone)')

    if all_atoms:
        
        for score in residue_score.values():
            for res,rescore in enumerate(score):
                if rescore>0:
                    resnum = str(res+1)
                    resnum.zfill(3)
                    view.add_ball_and_stick(f'{resnum} and (not hydrogen) and',opacity=0.75)
                    #view.add_ball_and_stick(f'{resnum}',opacity=0.5)

    # highlight selected features
    if relevant_features is not None:
        atom_ids = []
        #features = relevant_features[ classes_names[state] ]
        features = relevant_features[ next(iter(relevant_features)) ]
        
        if features_info is not None:
            for i, feature in enumerate(features):
                name = feature[2]
                print(name,features_info[name]['atoms'])
                atom_ids.append( features_info[name]['atoms'] )

            #colors = iter(['orange','green', 'purple', 'yellow', 'red'])
            colors=iter(paletteFessa)
            # loop over relevant features
            for ids in atom_ids:
                ids_string = [str(p) for p in ids]
                selection = '@'+','.join(ids_string)

                if color is None:
                    color = next(colors)

                if len(ids) == 2: # distance
                    #color = next(colors)
                    atom_pair = [ '@'+p for p in ids_string ]
                    view.add_distance(atom_pair=[atom_pair], color=color, label_visible=False)
                    #view.add_ball_and_stick(selection,color=color,opacity=0.75)
                elif len(ids) == 4: # angle
                    #color = 'green'
                    view.add_ball_and_stick(selection,color=color,opacity=0.75)

    return view
    