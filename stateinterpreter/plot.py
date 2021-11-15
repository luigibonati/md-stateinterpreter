import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
from matplotlib.gridspec import GridSpec
import nglview
import sys

__all__ = ["plot_states", "plot_cvpath", "plot_combination_cvs_relevant_features", "plot_cvs_relevant_features", "visualize_features"]

# aux function to compute basins mean
def compute_basin_mean(df, basin, label_x, label_y):
    mx = df[df['basin'] == basin][label_x].mean()
    my = df[df['basin'] == basin][label_y].mean()
    return mx,my

def plot_cvpath(classifier, reg, state_names=None, suptitle=None):
    assert classifier._computed, "You have to run Classifier.compute first."
    reg_idx = classifier._closest_reg_idx(reg)
    n_basins = classifier._coeffs.shape[1]

    if not state_names:
        state_names = [f'State {idx}' for idx in classifier.classes_labels[reg_idx]]
    assert len(state_names) == n_basins, "The length of state_names do not match the actual states."

    rows = np.int(np.ceil((n_basins + 1)/3))
    fig = plt.figure(constrained_layout=True, figsize=(9,3*rows))
    gs = GridSpec(rows, 3, figure=fig)

    axes = []
    for basin_idx in range(n_basins):
        axes.append(fig.add_subplot(gs[np.unravel_index(basin_idx, (rows, 3))]))
    axes.append(fig.add_subplot(gs[np.unravel_index(n_basins, (rows, 3))]))
    
    
    if suptitle:
        fig.suptitle(suptitle)
    for idx in range(n_basins):
        ax = axes[idx]
        _cfs = classifier._coeffs[:,idx,:]
        killer = np.abs(np.sum(_cfs, axis=0)) >= 1e-8
        ax.plot(np.log10(classifier._reg), _cfs[:,killer], 'k-')
        ax.axvline(x = np.log10(classifier._reg[reg_idx]), color='r', linewidth=0.75)
        ax.set_xlim(np.log10(classifier._reg[0]), np.log10(classifier._reg[-1]))
        ax.set_xlabel(r"$\log_{10}(\lambda)$")
        ax.set_title(state_names[idx])
        
    ax = axes[-1]
    ax.plot(np.log10(classifier._reg), classifier._crossval, 'k-')
    ax.axvline(x = np.log10(classifier._reg[reg_idx]), color='r', linewidth=0.75)
    ax.set_xlim(np.log10(classifier._reg[0]), np.log10(classifier._reg[-1]))
    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_title("Score")
    return (fig, axes)


def plot_states(data, fes_isolines = False, n_iso_fes = 9, ev_iso_labels = 2, save_folder=None):
    basins = data.basins['basin'].unique()
    n_basins = data.n_basins

    # hexbin plot of tica components 
    n_pairs = sum(1 for _ in itertools.combinations(data.selected_cvs, 2))

    fig, axs = plt.subplots(1,n_pairs,figsize=(6*n_pairs,5.5),dpi=100)
    idx_list = list(range(len(data.selected_cvs)))

    for k,(x_idx,y_idx) in enumerate(itertools.combinations(idx_list, 2)):
        label_x = data.selected_cvs[x_idx]
        label_y = data.selected_cvs[y_idx]
        # select ax
        ax = axs[k] if n_pairs > 1 else axs

        # FES isolines (if 2D)
        if fes_isolines: 
            if len(data.selected_cvs) == 2:
                cmap = matplotlib.cm.get_cmap('Greys_r', n_iso_fes )
                color_list = [cmap((i+1)/(n_iso_fes+3)) for i in range(n_iso_fes)]

                nx,ny=50,50
                xx, yy = np.meshgrid(np.linspace(data.bounds[0][0],data.bounds[0][1],nx),
                                    np.linspace(data.bounds[1][0],data.bounds[1][1],ny))
                fes = np.zeros_like(xx)
                for i in range(nx):
                    for j in range(ny):
                        fes[j,i] = data.fes( np.stack([xx[j,i], yy[j,i]]) )
                fes /= data.kbt
                fes -= np.min(fes)
                CS = ax.contour(xx, yy, fes, levels=np.linspace(0,n_iso_fes-1,n_iso_fes), colors = color_list) #colors='dimgrey')
                ax.clabel(CS, CS.levels[::ev_iso_labels], fmt = lambda x: str(int(x))+' kT', inline=True, fontsize=10)
                #ax.clabel(CS, CS.levels[::ev_iso_labels], inline=True, fontsize=10)
            else:
                raise NotImplementedError('Isolines are implemented only for 2D FES.')

        # hexbin plot
        x = data.colvar[label_x]
        y = data.colvar[label_y]
        z = data.basins['basin']
        sel = data.basins['selection']

        cmap_name = 'Set2'
        cmap = matplotlib.cm.get_cmap(cmap_name, n_basins)
        color_list = [cmap(i/(n_basins)) for i in range(n_basins)] 

        #cmap = matplotlib.cm.get_cmap('Set2', data.n_basins) 
        #norm = matplotlib.colors.BoundaryNorm(  np.linspace(0, data.n_basins, data.n_basins+1), cmap.N)
        
        ax.hexbin(x,y,C=z,cmap=cmap_name,alpha=0.3)#,norm=norm)
        ax.hexbin(x[sel],y[sel],C=z[sel],cmap=cmap_name)#,norm=norm)
        #cbar = plt.colorbar(pp,ax=ax)
        
        ax.set_title('Metastable states identification')
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        #df.plot.hexbin(label_x,label_y,C='basin',cmap='Set2',ax=ax,alpha=0.2,colorbar=False)
        #pp = df[df['selection']].plot.hexbin(label_x,label_y,C='basin',cmap='Set2',ax=ax)
        
        #add basins labels
        for b in basins:
            mx,my = data.minima[b][x_idx], data.minima[b][y_idx]  #compute_basin_mean(df,b,label_x,label_y)
            ax.scatter(mx,my,color='w',s=300,alpha=0.7)
            text = ax.text(mx, my, b, ha="center", va="center", 
                        color='k', fontsize='large')

    if save_folder is not None:
        plt.savefig(save_folder+'states.png',bbox_inches='tight')
    plt.tight_layout()

def plot_combination_cvs_relevant_features(df, selected_cvs, relevant_features, save_folder=None,file_prefix='linear'):
    
    added_columns = False
    #Handle quadratic kernels
    for _state in relevant_features:
        for _feat_tuple in _state:
            feature = _feat_tuple[2]
            if "||" in feature:
                if feature not in df.columns:
                    added_columns = True
                    i, j = feature.split(' || ')
                    feat_ij = df[i].values * df[j].values
                    df[feature] = feat_ij
    if added_columns:
        print("Warning: detected quadratic kenel features, added quadratic features to the input dataframe", file=sys.stderr)

    pairs = itertools.combinations(selected_cvs, 2)
    n_pairs = sum(1 for _ in pairs)

    for k,(cv_x,cv_y) in  enumerate(itertools.combinations(selected_cvs, 2)):

        plot_cvs_relevant_features(df, cv_x, cv_y, relevant_features, max_nfeat = 3)

        if save_folder is not None:
            plt.savefig(save_folder+file_prefix+f'-relevant_feats{k+1 if n_pairs > 1 else None}.png',
                        facecolor='w', 
                        transparent=False,
                        bbox_inches='tight')

def plot_cvs_relevant_features(df, cv_x, cv_y, relevant_feat, max_nfeat = 3):
    # retrieve basins
    basins = df['basin'].unique()
    n_basins = len(relevant_feat)

    fig, axs = plt.subplots(n_basins,max_nfeat,figsize=(6 * max_nfeat, 5* n_basins),dpi=100, )
                            #sharex=True, sharey=True)

    # for each state ...
    for i, feat_list in enumerate(relevant_feat):
        #This is wrong it should be modified with classes labels
        state = i
        # ... color with the corresponding features ...
        for j,feat_array in enumerate(feat_list):
            # ... up to max_nfeat plot per state
            if j < max_nfeat:
                feat = feat_array[2]
                importance = feat_array[1]
                ax = axs[i,j]
                pp = df[df['selection']==1].plot.hexbin(cv_x,cv_y,C=feat,cmap='coolwarm',ax=ax)
                #set title
                ax.set_title(f'[state {state}] {feat} - {np.round(importance*100)}%')
                #add basins labels
                for b in basins:
                    mx,my = compute_basin_mean(df,b,cv_x,cv_y)
                    bcolor = 'k' if b == state else 'w'
                    fcolor = 'w' if b == state else 'k'            
                    ax.scatter(mx,my,color=bcolor,s=250,alpha=0.7)
                    text = ax.text(mx, my, b, ha="center", va="center", 
                                color=fcolor, fontsize='large')
        
        #set labels
        for ax in plt.gcf().axes:
            try:
                ax.label_outer()
            except:
                pass

        #disable unused axis
        for j in range(len(feat_list),max_nfeat):
            axs[i,j].axis('off')

    plt.tight_layout()

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
            name = feature[1]
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
