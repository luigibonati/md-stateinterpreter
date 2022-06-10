import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import sys
from .._configs import *
from .numerical_utils import gaussian_kde

if __useTeX__:
    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "cm",
        #"font.family": "serif",
        #"font.serif": ["Computer Modern Roman"]
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"]
    })

__all__ = ["plot_states", "plot_regularization_path", "plot_classifier_complexity_vs_accuracy", "plot_combination_states_features", "plot_states_features", "plot_histogram_features" ]

##########################################################################
## FESSA COLOR PALETTE
#  https://github.com/luigibonati/fessa-color-palette/blob/master/fessa.py
##########################################################################

from matplotlib.colors import LinearSegmentedColormap, ColorConverter
from matplotlib.cm import register_cmap

paletteFessa = [
    '#1F3B73', # dark-blue
    '#2F9294', # green-blue
    '#50B28D', # green
    '#A7D655', # pisello
    '#FFE03E', # yellow
    '#FFA955', # orange
    '#D6573B', # red
]

cm_fessa = LinearSegmentedColormap.from_list('fessa', paletteFessa)
register_cmap(cmap=cm_fessa)
register_cmap(cmap=cm_fessa.reversed())

for i in range(len(paletteFessa)):
    ColorConverter.colors[f'fessa{i}'] = paletteFessa[i]

### To set it as default
# import fessa
# plt.set_cmap('fessa')
### or the reversed one
# plt.set_cmap('fessa_r')
### For contour plots
# plt.contourf(X, Y, Z, cmap='fessa')
### For standard plots
# plt.plot(x, y, color='fessa0')

##########################################################################

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

def plot_classifier_complexity_vs_accuracy(classifier, feature_mode = False, ax = None):
    assert classifier._computed, "You have to run Classifier.compute first."
    num_groups = []
    for reg in classifier._reg:
        selected = classifier._get_selected(reg, feature_mode=feature_mode)
        unique_idxs = set()
        for state in selected.values():
            for data in state:
                unique_idxs.add(data[0])
        num_groups.append(len(unique_idxs)) 
    
    if ax is not None:
        ax1 = ax
    else:
        fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax2.grid(alpha=0.3)
    ax2.plot(np.log10(classifier._reg), num_groups, '--', color='fessa1')
    ax1.plot(np.log10(classifier._reg), classifier._crossval, '-', color='fessa0')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel(r"$\log_{10}(\lambda)$")
    ax1.set_ylabel('Accuracy', color='fessa0')
    ax1.set_ylim(0,1.1)
    desc = "Groups" if classifier._groups is not None else "Features"
    ax2.set_ylabel(f'Number of {desc}', color='fessa1')
    ax1.set_xmargin(0)

    if ax is not None:
        return (ax1, ax2)
    else:
        return fig, (ax1, ax2)

def plot_states(colvar, state_labels, selected_cvs, fes_isolines = False, n_iso_fes = 9, ev_iso_labels = 2, alpha=0.3, cmap_name = 'Set2', save_folder=None, axs = None, **kde_kwargs):
    states = state_labels['labels'].unique()
    n_states = len(states)

    # hexbin plot of tica components 
    idxs_pairs = [p for p in combinations(np.arange(len(selected_cvs)), 2)]
    n_pairs = len(idxs_pairs)

    if axs is None:
        fig, axs = plt.subplots(1,n_pairs,figsize=(4.8*n_pairs,4), dpi=100)

    for k, (x_idx,y_idx) in enumerate(idxs_pairs):
        
        label_x = selected_cvs[x_idx]
        label_y = selected_cvs[y_idx]
        # select ax
        ax = axs[k] if n_pairs > 1 else axs

        # FES isolines (if 2D)
        if fes_isolines:
            #logweights = None
            #bw_method = 0.15

            num_samples = 100

            cmap = matplotlib.cm.get_cmap('Greys_r', n_iso_fes)
            color_list = [cmap((i+1)/(n_iso_fes+3)) for i in range(n_iso_fes)]

            empirical_centers = colvar[[label_x,label_y]].to_numpy()
            KDE = gaussian_kde(empirical_centers,**kde_kwargs)

            bounds = [(x.min(), x.max()) for x in KDE.dataset.T]
            mesh = np.meshgrid(*[np.linspace(b[0], b[1], num_samples) for b in bounds])

            positions = np.vstack([g.ravel() for g in mesh]).T
            fes = -KDE.logpdf(positions).reshape(num_samples,num_samples)
            fes -= fes.min()

            CS = ax.contour(*mesh, fes, levels=np.linspace(0,n_iso_fes-1,n_iso_fes), colors = color_list)
            ax.clabel(CS, CS.levels[::ev_iso_labels], fmt = lambda x: str(int(x))+ r'$k_{{\rm B}}T$', inline=True, fontsize=8)

        # Hexbin plot
        x = colvar[label_x].values
        y = colvar[label_y].values
        z = state_labels['labels'].values
        sel = state_labels['selection'].values
        not_sel = np.logical_not(sel)

        cmap = matplotlib.cm.get_cmap(cmap_name, n_states)
        color_list = [cmap(i/(n_states)) for i in range(n_states)] 
      
        ax.hexbin(x[not_sel],y[not_sel],C=z[not_sel],cmap=cmap_name,alpha=alpha)
        ax.hexbin(x[sel],y[sel],C=z[sel],cmap=cmap_name)
     
        if axs is None:
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
    
    if axs is None:
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


def plot_histogram_features(descriptors,states_labels,classes_names,relevant_feat, hist_offset = -0.2, n_bins = 50, ylog = False, axs = None, height=1, width=6, colors=None):
    #TODO MOVE PLOT KEYWORDS inTO DICT

    features_per_class = [ len(feat_list) for feat_list in relevant_feat.values() ]

    added_columns = False
    #Handle quadratic kernels
    for _state in relevant_feat.values():
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

    if axs is None:
        tight=True
        fig,axs = plt.subplots(len(relevant_feat), 1,
                            figsize=(width, sum(features_per_class)*height ),
                            gridspec_kw={'height_ratios': features_per_class})
        if len(relevant_feat) == 1:
            axs = [axs]
    else:
        tight=False

    #for b, (basin, basin_name) in enumerate( classes_names.items() ):
    for b, (basin_name,feat_list) in enumerate(relevant_feat.items()) :
        def get_key(dict, val):
            for key, value in dict.items():
                if val == value:
                    return key

        basin = get_key(classes_names,basin_name)

        #feat_list = relevant_feat[basin_name]

        #fig,ax = plt.subplots( figsize = (4,0.5*len(feat_list)) )
        ax = axs[b]
        feature_labels = []
        for h, feature in enumerate( feat_list[::-1] ):
            feature_name = feature[2]
            if (__useTeX__) and ('_' in feature_name):
                feature_label = feature_name.replace('_','\_')
            else:
                feature_label = feature_name
            feature_labels.append(feature_label)
            #coordinate = descriptors[feature_name]
            #hist, edges = np.histogram(coordinate, bins=n_bins)
            for i in classes_names.keys():
                x_i = descriptors[ ( states_labels['labels'] == classes_names[i] ) & ( states_labels['selection'] ) ][feature_name]
                hist, edges = np.histogram(x_i, bins=n_bins)
                if not ylog:
                    y = hist / hist.max()
                else:
                    y = np.zeros_like(hist) + np.NaN
                    pos_idx = hist > 0
                    y[pos_idx] = np.log(hist[pos_idx]) / np.log(hist[pos_idx]).max()
                if colors is not None:
                    color = colors[i]
                else:
                    color = f'fessa{6-i}' #'tab:red' if basin == i else 'dimgray'
                ax.plot(edges[:-1], y + h + hist_offset,color=color)
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

    if tight:
        plt.tight_layout()

def plot_fes(cv,bandwidth,states_labels=None,logweights=None,kBT=2.5,cv_list=None,states_subset=None,num_samples=100,ax=None,prefix_label="",colors=None):
    
    if cv_list is not None:
        cv = cv[cv_list]

    empirical_centers = cv.to_numpy()
    KDE = gaussian_kde(empirical_centers,bandwidth,logweights)

    bounds = [(x.min(), x.max()) for x in KDE.dataset.T]
    mesh = np.meshgrid(*[np.linspace(b[0], b[1], num_samples) for b in bounds])

    positions = np.vstack([g.ravel() for g in mesh]).T
    fes = -kBT*KDE.logpdf(positions)
    fes -= fes.min()
    if ax is None:
        _ , ax = plt.subplots()
    ax.plot(mesh[0],fes/kBT,color='dimgrey',linewidth=1.5)
    ax.set_xlabel(cv.columns.values[0])
    ax.set_ylabel('FES [$k_B$T]')
    ax.set_xlim(bounds[0][0],bounds[0][1])
    ax.set_ylim(0,)
    
    if states_labels is not None:
        if states_subset is not None:
            labels = states_subset
        else:
            labels = sorted(states_labels['labels'].unique())
        _pruned_states_labels = states_labels[states_labels['labels'] != 'undefined']
        for i,label in enumerate(labels):
            if label != 'undefined':
                mask = ( _pruned_states_labels['labels'] == label ) & (_pruned_states_labels['selection'] == True )
                
                Min = cv[mask].min().values[0] 
                Max = cv[mask].max().values[0] 
                if colors is not None:
                    color = colors[i]
                else:
                    color = f'fessa{6-i}'
                ax.axvspan(Min,Max, alpha=0.5, color=color)
                ax.text((Max+Min)/2,5,prefix_label+str(label),fontsize='medium',ha='center')


def plot_fes_2d(colvar, state_labels, selected_cvs, n_iso_fes = 10, ev_iso_labels = 2, save_folder=None, ax = None, xlim=[-1,1.], ylim=[-1,1.], label_names = None, label_colors= None, **kde_kwargs):
    states = state_labels['labels'].unique()

    xlim=[-1,1.05]
    ylim=[-1,1.05]

    label_x = selected_cvs[0]
    label_y = selected_cvs[1]

    # FES ISOLINES
    num_samples = 100

    cmap = matplotlib.cm.get_cmap('Greys_r', n_iso_fes)
    color_list = [cmap((i+1)/(n_iso_fes+3)) for i in range(n_iso_fes)]

    empirical_centers = colvar[[label_x,label_y]].to_numpy()
    KDE = gaussian_kde(empirical_centers,**kde_kwargs)

    bounds = [(x.min(), x.max()) for x in KDE.dataset.T]
    mesh = np.meshgrid(*[np.linspace(b[0], b[1], num_samples) for b in bounds])

    positions = np.vstack([g.ravel() for g in mesh]).T
    fes = -KDE.logpdf(positions).reshape(num_samples,num_samples)
    fes -= fes.min()
    if ax is None:
        _, ax = plt.subplots()
    CS = ax.contour(*mesh, fes, levels=np.linspace(0,n_iso_fes-1,n_iso_fes), colors = color_list)
    ax.clabel(CS, CS.levels[::ev_iso_labels], fmt = lambda x: str(int(x))+ r'$k_{{\rm B}}T$', inline=True, fontsize=8)

    # Add basins labels
    x = colvar[label_x]
    y = colvar[label_y]
    z = state_labels['labels']
    sel = state_labels['selection']

    if label_names is None:
        label = states
    else:
        label = label_names

    if label_colors is None:
        color=[paletteFessa[i] for i in range(len(states))]
    else:
        color = label_colors

    for b in states:
        mask = np.logical_and(sel, z == b)
        mx,my = np.average(x[mask],weights=np.exp(kde_kwargs['logweights'][mask])), np.average(y[mask],weights=np.exp(kde_kwargs['logweights'][mask]))
        #ax.scatter(mx,my,color=color_list[b],s=300,alpha=1)
        
        #if b>0:
        #    ax.scatter(mx,my,color=color[b],s=550,alpha=0.5,facecolors=None,edgecolors=paletteFessa[6])
        ax.scatter(mx,my,color=color[b],s=450,alpha=0.5,edgecolors=None)
        _ = ax.text(mx, my, label[b], ha="center", va="center", color='k', fontsize='large')

    ax.set_xlabel(selected_cvs[0])
    ax.set_ylabel(selected_cvs[1])