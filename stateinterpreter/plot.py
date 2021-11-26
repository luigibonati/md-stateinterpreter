import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import nglview
import sys
from ._configs import *

if __useTeX__:
    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })

__all__ = ["plot_states", "plot_regularization_path", "plot_classifier_complexity_vs_accuracy", "plot_combination_cvs_relevant_features", "plot_cvs_relevant_features", "visualize_features", "visualize_residues"]

# aux function to compute basins mean
def compute_basin_mean(df, basin, label_x, label_y):
    mx = df[df['basin'] == basin][label_x].mean()
    my = df[df['basin'] == basin][label_y].mean()
    return mx,my

def plot_regularization_path(classifier, reg):
    assert classifier._computed, "You have to run Classifier.compute first."
    reg_idx = classifier._closest_reg_idx(reg)
    n_basins = classifier._coeffs.shape[1]

    rows = np.int(np.ceil((n_basins + 1)/3))
    fig = plt.figure(constrained_layout=True, figsize=(8,2.5*rows))
    gs = GridSpec(rows, 3, figure=fig)

    axes = []
    for basin_idx in range(n_basins):
        axes.append(fig.add_subplot(gs[np.unravel_index(basin_idx, (rows, 3))]))
    axes.append(fig.add_subplot(gs[np.unravel_index(n_basins, (rows, 3))]))
    
    fig.suptitle(r"Regularization paths")

    for idx, state_idx in enumerate(classifier._classes_labels[reg_idx]):
        ax = axes[idx]
        _cfs = classifier._coeffs[:,idx,:]
        killer = np.abs(np.sum(_cfs, axis=0)) >= __EPS__
        ax.plot(np.log10(classifier._reg), _cfs[:,killer], 'k-')
        ax.axvline(x = np.log10(classifier._reg[reg_idx]), color='tomato', linewidth=0.75)
        ax.set_xmargin(0)
        ax.set_xlabel(r"$\log_{10}(\lambda)$")
        ax.set_title(classifier.classes[state_idx])
        
    ax = axes[-1]
    ax.plot(np.log10(classifier._reg), classifier._crossval, 'k-')
    ax.axvline(x = np.log10(classifier._reg[reg_idx]), color='r', linewidth=0.75)
    ax.set_xmargin(0)
    ax.set_ylim(0,1.1)
    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_title(r"Accuracy")
    return fig, axes

def plot_classifier_complexity_vs_accuracy(classifier, feature_mode = False):
    assert classifier._computed, "You have to run Classifier.compute first."
    num_groups = []
    for reg in classifier._reg:
        selected = classifier._get_selected(reg, feature_mode=feature_mode)
        unique_idxs = set()
        for state in selected.values():
            for data in state:
                unique_idxs.add(data[0])
        num_groups.append(len(unique_idxs)) 
    
    fig, ax1 = plt.subplots(figsize=(3,2))
    ax2 = ax1.twinx()
    ax2.grid(alpha=0.3)
    ax2.plot(np.log10(classifier._reg), num_groups, '-', color='steelblue')
    ax1.plot(np.log10(classifier._reg), classifier._crossval, '-', color='tomato')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel(r"$\log_{10}(\lambda)$")
    ax1.set_ylabel('Accuracy', color='r')
    ax1.set_ylim(0,1.1)
    desc = "Groups" if classifier._groups is not None else "Features"
    ax2.set_ylabel(f'Number of {desc}', color='b')
    ax1.set_xmargin(0)
    return fig, (ax1, ax2)

def plot_states(colvar, state_labels, selected_cvs, fes_isolines = False, n_iso_fes = 9, ev_iso_labels = 2, save_folder=None):
    states = state_labels['labels'].unique()
    n_states = len(states)

    # hexbin plot of tica components 
    idxs_pairs = [p for p in combinations(np.arange(len(selected_cvs)), 2)]
    n_pairs = len(idxs_pairs)

    fig, axs = plt.subplots(1,n_pairs,figsize=(4.8*n_pairs,4), dpi=100)

    for k, (x_idx,y_idx) in enumerate(idxs_pairs):
        label_x = selected_cvs[x_idx]
        label_y = selected_cvs[y_idx]
        # select ax
        ax = axs[k] if n_pairs > 1 else axs

        # FES isolines (if 2D)
        if fes_isolines:
            raise NotImplementedError('Isolines not implemented.')
            '''
            if len(selected_cvs) == 2:
                cmap = matplotlib.cm.get_cmap('Greys_r', n_iso_fes)
                color_list = [cmap((i+1)/(n_iso_fes+3)) for i in range(n_iso_fes)]
                num_samples = 100
                bounds = [(x.min(), x.max()) for x in data.KDE.dataset.T]
                mesh = np.meshgrid(*[np.linspace(b[0], b[1], num_samples) for b in bounds])
                positions = np.vstack([g.ravel() for g in mesh]).T
                fes = -data.KDE.logpdf(positions).reshape(num_samples,num_samples)
                fes -= fes.min()
                CS = ax.contour(*mesh, fes, levels=np.linspace(0,n_iso_fes-1,n_iso_fes), colors = color_list)
                ax.clabel(CS, CS.levels[::ev_iso_labels], fmt = lambda x: str(int(x))+ r'$k_{{\rm B}}T$', inline=True, fontsize=10)
            else:
                raise NotImplementedError('Isolines are implemented only for 2D FES.')
            ''' 

        # Hexbin plot
        x = colvar[label_x]
        y = colvar[label_y]
        z = state_labels['labels']
        sel = state_labels['selection']
        not_sel = np.logical_not(sel)

        cmap_name = 'Set2'
        cmap = matplotlib.cm.get_cmap(cmap_name, n_states)
        color_list = [cmap(i/(n_states)) for i in range(n_states)] 
      
        ax.hexbin(x[not_sel],y[not_sel],C=z[not_sel],cmap=cmap_name,alpha=0.3)
        ax.hexbin(x[sel],y[sel],C=z[sel],cmap=cmap_name)
     
        ax.set_title('Metastable states identification')
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
    
        #Add basins labels
        for b in states:
            mask = np.logical_and(sel, z == b)
            #If weighted not ok but functional
            mx,my = np.mean(x[mask]), np.mean(y[mask])
            ax.scatter(mx,my,color='w',s=300,alpha=0.7)
            _ = ax.text(mx, my, b, ha="center", va="center", color='k', fontsize='large')

    if save_folder is not None:
        plt.savefig(save_folder+'states.pdf',bbox_inches='tight')
    plt.tight_layout()
    return fig, axs

def plot_combination_states_relevant_features(colvar, descriptors, selected_cvs, relevant_features, state_labels = None, save_folder=None, file_prefix='linear'):
    
    if len(selected_cvs) < 2: 
        raise NotImplementedError('This plot is available only when selecting 2 or more CVs.')

    added_columns = False
    #Handle quadratic kernels
    for _state in relevant_features.values():
        for _feat_tuple in _state:
            feature = _feat_tuple[2]
            if "||" in feature:
                if feature not in descriptors.columns:
                    added_columns = True
                    i, j = feature.split(' || ')
                    feat_ij = descriptors[i].values * descriptors[j].values
                    descriptors[feature] = feat_ij
    if added_columns:
        print("Warning: detected quadratic kenel features, added quadratic features to the input dataframe", file=sys.stderr)

    pairs = combinations(selected_cvs, 2)
    n_pairs = sum(1 for _ in pairs)

    for k,(label_x,label_y) in  enumerate(combinations(selected_cvs, 2)):
        cv_x = colvar[label_x].values
        cv_y = colvar[label_y].values
        plot_states_relevant_features(cv_x, cv_y, descriptors, relevant_features, state_labels=state_labels, max_nfeat = 3)

        if save_folder is not None:
            plt.savefig(save_folder+file_prefix+f'-relevant_feats{k+1 if n_pairs > 1 else None}.png',
                        facecolor='w', 
                        transparent=False,
                        bbox_inches='tight')

def plot_states_relevant_features(cv_x, cv_y, descriptors, relevant_feat, state_labels = None, max_nfeat = 3):
    n_basins = len(relevant_feat)

    # if state_labels are given plot only selection
    if state_labels is not None:
        mask = state_labels['selection']
        cv_x = cv_x[mask]
        cv_y = cv_y[mask]
        descriptors = descriptors[mask]
        state_labels = state_labels[mask]

    fig, axs = plt.subplots(n_basins,max_nfeat,figsize=(6 * max_nfeat, 5* n_basins),dpi=72)
    # for each state ...
    for i, (state_name, feat_list) in enumerate( relevant_feat.items() ):
        state = i
        # ... color with the corresponding features ...
        for j,feat_array in enumerate(feat_list):
            # ... up to max_nfeat plot per state
            if j < max_nfeat:
                feat = feat_array[2]
                importance = feat_array[1]
                ax = axs[i,j]
                #pp = df[df['selection']==1].plot.hexbin(cv_x,cv_y,C=feat,cmap='coolwarm',ax=ax)
                pp = ax.hexbin(cv_x,cv_y,C=descriptors[feat],cmap='coolwarm')
                #set title
                if '_' in feat:
                    feat = feat.replace('_','\_')
                ax.set_title(f'[{state}: {state_name}] {feat} - {np.round(importance*100)}%')
                #add basins labels if given
                if state_labels is not None:
                    states = state_labels['labels'].unique()
                    z = state_labels['labels']
                    #Add basins labels
                    for b in states:
                        #mask = np.logical_and(sel, z == b)
                        mask = ( z == b ) 
                        #If weighted not ok but functional
                        mx,my = np.mean(cv_x[mask]), np.mean(cv_y[mask])
                        ax.scatter(mx,my,color='w',s=300,alpha=0.7)
                        _ = ax.text(mx, my, b, ha="center", va="center", color='k', fontsize='large')


def visualize_features(data,relevant_features,state=0,n_feat_per_state=3):
    
    # sample one frame per state
    frames = [data.basins[data.basins['basin'] == i ].sample(1).index.values[0] for i in range(data.n_basins) ]
    traj = data.traj[frames]
    traj.superpose(traj[state])

    # find atom ids of relevant features
    atom_ids = []

    features = relevant_features[state]
    for i, feature in enumerate(features):
        if i < n_feat_per_state:
            name = feature[2]
            atom_ids.append( data.descriptors_ids[name] )

    # set up visualization
    view = nglview.show_mdtraj(traj)
    #view.frame = state
    view.clear_representations()

    # draw backbone + transparent sidechain
    view.add_licorice('(not hydrogen)',opacity=0.35)
    view.add_licorice('(backbone) and (not hydrogen)',opacity=0.85)

    # loop over relevant features
    for ids in atom_ids:
        ids_string = [str(p) for p in ids]
        selection = '@'+','.join(ids_string)

        if len(ids) == 2: # distance
            color = 'orange'
            atom_pair = [ '@'+p for p in ids_string ]
            view.add_distance(atom_pair=[atom_pair], color=color, label_visible=False)
            view.add_ball_and_stick(selection,color=color,opacity=0.75)
        elif len(ids) == 4: # angle
            color = 'green'
            view.add_ball_and_stick(selection,color=color,opacity=0.75)

    return view

def visualize_residues(data, residue_score, representation = 'licorice', palette = 'Reds'):
    # sample one frame per state
    frames = [data.basins[data.basins['basin'] == i ].sample(1).index.values[0] for i in range(data.n_basins) ]
    traj = data.traj[frames]
    traj.superpose(traj[0])

    view = nglview.show_mdtraj(traj, default=False)

    if representation == 'licorice' :
        view.add_licorice('all')
    elif representation == 'cartoon':
        view.add_cartoon('protein')
    elif representation == 'ball_and_stick':
        view.add_ball_and_stick('all')

    # get color palette
    cmap = cm.get_cmap(palette, 11)
    palette = [matplotlib.colors.rgb2hex( cmap(i) ) for i in range(cmap.N)]

    # transform score in colors
    residue_colors = {}
    for state in range(data.n_basins):
        colors = []
        if state in residue_score.keys():
            for i in residue_score[state]:
                col = int (i*10 / 1)
                colors.append( palette[col] ) 
        else:
            for j in range( data.traj.n_residues ):
                colors.append( palette[0] )
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