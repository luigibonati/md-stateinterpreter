import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import sys
from ._configs import *

if __useTeX__:
    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })

__all__ = ["plot_states", "plot_regularization_path", "plot_classifier_complexity_vs_accuracy", "plot_combination_states_features", "plot_states_features", "plot_histogram_features" ]

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

def plot_combination_states_features(colvar, descriptors, selected_cvs, relevant_features, state_labels = None, save_folder=None, file_prefix='linear'):
    
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
        plot_states_features(cv_x, cv_y, descriptors, relevant_features, state_labels=state_labels, max_nfeat = 3)

        if save_folder is not None:
            plt.savefig(save_folder+file_prefix+f'-relevant_feats{k+1 if n_pairs > 1 else None}.png',
                        facecolor='w', 
                        transparent=False,
                        bbox_inches='tight')

def plot_states_features(cv_x, cv_y, descriptors, relevant_feat, state_labels = None, max_nfeat = 3):
    n_basins = len(relevant_feat)

    # if state_labels are given plot only selection
    if state_labels is not None:
        mask = state_labels['selection']
        cv_x = cv_x[mask]
        cv_y = cv_y[mask]
        descriptors = descriptors[mask]
        state_labels = state_labels[mask]

    fig, axs = plt.subplots(n_basins,max_nfeat,figsize=(4 * max_nfeat, 3.5* n_basins), 
                            sharex=True, sharey=True)
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
                if (__useTeX__) and ('_' in feat):
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


def plot_histogram_features(descriptors,states_labels,classes_names,relevant_feat,hist_offset = -0.2,n_bins = 50,ylog = False, height=0.75,width=6):
    #TODO MOVE PLOT KEYWORDS inTO DICT

    features_per_class = [len(feat_list) for feat_list in relevant_feat.values()]
    fig,axs = plt.subplots(len(classes_names), 1,
                            figsize=(width, sum(features_per_class)*height ),
                            gridspec_kw={'height_ratios': features_per_class})

    for b, (basin, basin_name) in enumerate( classes_names.items() ):
        feat_list = relevant_feat[basin_name]

        #fig,ax = plt.subplots( figsize = (4,0.5*len(feat_list)) )
        ax = axs[b]
        feature_labels = []
        for h, feature in enumerate( feat_list ):
            feature_name = feature[2]
            if (__useTeX__) and ('_' in feature_name):
                feature_label = feature_name.replace('_','\_')
            else:
                feature_label = feature_name
            feature_labels.append(feature_label)
            #coordinate = descriptors[feature_name]
            #hist, edges = np.histogram(coordinate, bins=n_bins)
            for i in classes_names.keys():
                x_i = descriptors[ ( states_labels['labels'] == i ) & ( states_labels['selection'] ) ][feature_name]
                hist, edges = np.histogram(x_i, bins=n_bins)
                if not ylog:
                    y = hist / hist.max()
                else:
                    y = np.zeros_like(hist) + np.NaN
                    pos_idx = hist > 0
                    y[pos_idx] = np.log(hist[pos_idx]) / np.log(hist[pos_idx]).max()
                color = 'tab:red' if basin == i else 'dimgray'
                ax.fill_between(edges[:-1], y + h + hist_offset, y2=h + hist_offset, color=color, alpha=0.5) #, **kwargs)

            ax.axhline(y=h + hist_offset, xmin=0, xmax=1, color='k', linewidth=.2)
        ax.set_ylim(hist_offset, h + hist_offset + 1)

        # formatting
        if feature_labels is None:
            feature_labels = [str(n) for n in range(len(feat_list))]
            ax.set_ylabel('Feature histograms')

        ax.set_yticks(np.array(range(len(feature_labels))) + .3)
        ax.set_yticklabels(feature_labels)
        #ax.set_xlabel('Feature values')
        ax.set_title(f'{basin}: {basin_name}')

    plt.tight_layout()