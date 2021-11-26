import nglview
import matplotlib
import matplotlib.cm as cm
import numpy as np

def visualize_features(trajectory, states_labels, classes_names, relevant_features, feats_info, state = 0, n_feat_per_state=3, representation = 'licorice'):
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
    # get relevant features with feature_mode=True
    relevant_feat = classifier._get_selected(reg,feature_mode=True)
    
    residue_score = {} 
    #loop over states
    for state in relevant_feat.keys():
        score = np.zeros(n_residues)
        basin_data = relevant_feat[state]
        # loop over relevant features per state
        for feat in basin_data:
            # feature name 
            feat_name = feat[2]
            # get residue from group
            resname = feats_info[feat_name]['group']
            #resname = feat[2].split(' ')[-1]
            for res in resname.split('_'):
                resnum = int ( ''.join([n for n in resname if n.isdigit()]) )
                # increase score with weight
                score [resnum - 1] += feat[1]
        residue_score[state] = score

    return residue_score

def visualize_residue_score(trajectory, states_labels, classes_names, residue_score, representation = 'licorice', palette = 'Reds'):
    # sample one frame per state
    frames = [states_labels [( states_labels['labels'] == i ) & ( states_labels['selection'] ) ].sample(1).index.values[0] for i in classes_names.keys() ]
    traj = trajectory[frames]
    traj.superpose(traj[0])

    view = nglview.show_mdtraj(traj, default=False)

    if representation == 'licorice' :
        view.add_licorice('(not hydrogen)')
    elif representation == 'cartoon' :
        view.add_cartoon('protein')
    elif representation == 'ball_and_stick' :
        view.add_ball_and_stick('(not hydrogen)')

    # get color palette
    cmap = cm.get_cmap(palette, 11)
    palette = [matplotlib.colors.rgb2hex( cmap(i) ) for i in range(cmap.N)]

    # transform score in colors
    residue_colors = {}
    for i, state in enumerate( classes_names.keys() ):
        colors = []
        for i in residue_score[ classes_names[state] ]:
            col = int (i*10)
            colors.append( palette[col] ) 
        residue_colors[state] = colors

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