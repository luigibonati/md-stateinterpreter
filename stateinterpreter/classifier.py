import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import concurrent.futures
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from matplotlib.gridspec import GridSpec
import scipy.sparse.linalg
import sys

class MD_Data:
    def __init__(self, dataframe):
        self._df = dataframe
        self.n_clusters = self._df['basin'].max() + 1
    def sample(self, n_configs, regex_filter = '.*'):
        features = self._df.filter(regex=regex_filter).columns.values
        config_list = []
        labels = []
        for basin in range(self.n_clusters):
            #select basin
            df = self._df.loc[ (self._df['basin'] == basin) & (self._df['selection'] == True)]
            #select descriptors and sample
            config_i = df.filter(regex=regex_filter).sample(n=n_configs).values
            config_list.append(config_i)
            labels.extend([basin]*n_configs)
        labels = np.array(labels, dtype=np.int_)
        configurations = np.vstack(config_list)
        return Sample(configurations, features, labels, scale=True)

    
    def sample_feature(self, feature_name, get_angle=False, n_configs=100):
        feat_info = parse_feature_name(feature_name)
        _data = []
        #Check if given feature is angle
        if len(feat_info) == 1:
            _is_sin = False
            _is_cos = False
            if feat_info[feature_name]['sin']== True:
                _is_sin = True
            if feat_info[feature_name]['cos']== True:
                _is_cos = True
            _is_angle = (_is_sin or _is_cos)
            if _is_angle:
                if not get_angle:
                    print("Given feature is an angle. If you want to parse the angle instead of the given feature use get_angle=True")
            if get_angle:
                if (_is_sin ==False) and (_is_cos ==False):
                    raise ValueError("Given feature is not an angle. Not with get_angle=True")
                
            for basin_idx in range(self.n_clusters):
                df = self._df.loc[ (self._df['basin'] == basin_idx) & (self._df['selection'] == True)]
                if _is_angle and get_angle:
                    if feat_info[feature_name]['sin']:
                        _sin = df.filter(regex=feature_name).sample(n=n_configs).values
                        _cos = df.filter(regex=feature_name.replace('sin_', 'cos_')).sample(n=n_configs).values
                        _data.append(np.arctan2(_sin, _cos))
                    elif feat_info[feature_name]['cos']:
                        _cos = df.filter(regex=feature_name).sample(n=n_configs).values
                        _sin = df.filter(regex=feature_name.replace('cos_', 'sin_')).sample(n=n_configs).values
                        _data.append(np.arctan2(_sin, _cos))
                elif feat_info[feature_name]['squared']:
                    _data.append((df.filter(regex=feature_name).sample(n=n_configs).values)**2)
                else: 
                    _data.append(df.filter(regex=feature_name).sample(n=n_configs).values)
        if len(feat_info) > 1:
            for basin_idx in range(self.n_clusters):
                df = self._df.loc[ (self._df['basin'] == basin_idx) & (self._df['selection'] == True)]
                _val = np.ones(n_configs)
                for feat in feat_info.keys():
                    _val *= df.filter(regex=feat).sample(n=n_configs).values
                _data.append(_val)
        return np.squeeze(np.asarray(_data))
    

class Sample:
    def __init__(self, configurations, features, labels, scale=False):
        self.unscaled_configurations = configurations
        self.configurations = configurations
        self.features = features
        self.labels = labels
        if scale:
            self.scale()
    def scale(self):
        self.scaler = StandardScaler(with_mean=True)
        self.scaler.fit(self.configurations)
        self.configurations = self.scaler.transform(self.unscaled_configurations)
    def train_test_dataset(self, **kwargs):
        return train_test_split(self.configurations, self.labels, **kwargs)
    @property
    def dataset(self):
        return [self.configurations, self.labels]

def quadratic_kernel_featuremap(X):
    n_pts, n_feats = X.shape
    n_feats +=1
    transformed_X = np.empty((n_pts, n_feats + n_feats*(n_feats - 1)//2), dtype=np.float_)
    X = np.c_[X, np.ones(n_pts)]
    
    def _compute_repr(x):
        mat = np.outer(x,x)
        diag = np.diag(mat)
        mat = (mat - np.diag(diag))*np.sqrt(2)
        off_diag = squareform(mat)
        return np.r_[diag, off_diag]

    for idx, x in enumerate(X):
        transformed_X[idx] = _compute_repr(x)
    return transformed_X

def decode_quadratic_features(idx, features_names):
    num_feats = features_names.shape[0]
    s = ''
    if idx < num_feats:
        s = f"{features_names[idx]} || {features_names[idx]}"
    elif idx > num_feats:
        rows, cols = np.triu_indices(num_feats + 1, k = 1)
        offset_idx  = idx -  num_feats - 1
        i, j = rows[offset_idx], cols[offset_idx]
        if i == num_feats:
            s = f"{features_names[j]}"
        elif j == num_feats:
            s = f"{features_names[i]}"
        else:
            s = f"{features_names[i]} || {features_names[j]}"
    return s

class CV_path():
    def __init__(self, dataset, features, quadratic_kernel=False):
        self._dset = dataset
        self._n_samples = dataset[0].shape[0]
        self._features = features
        self._quadratic_kernel = quadratic_kernel
    
    def compute(self, C_range, l1_ratio=None, multi_class="multinomial", **kwargs):
        '''If kwargs LASSO is defined use Lasso. Elasticnet otherwise.'''
        quadratic_kernel = self._quadratic_kernel
        train_in, val_in, train_out, val_out = self._dset
        if quadratic_kernel:
            train_in, val_in = quadratic_kernel_featuremap(train_in), quadratic_kernel_featuremap(val_in)
        _is_lasso = kwargs.get('LASSO', False)
        try:
            del kwargs['LASSO']
        except:
            pass

        if _is_lasso:
            penalty = 'l1' 
        else:
            penalty ='elasticnet'

        def _train_model(C):
            model = LogisticRegression(penalty=penalty, C=C, solver='saga', l1_ratio=l1_ratio, multi_class=multi_class, fit_intercept=False, **kwargs) 
            #Model Fit
            model.fit(train_in,train_out)
            score = model.score(val_in,val_out)
            return (C, model.coef_,score)
        
        path_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut = [executor.submit(_train_model, C) for C in C_range]
            for fut_result in concurrent.futures.as_completed(fut):
                path_data.append(fut_result.result())

        n_features = train_in.shape[1]
        n_C = C_range.shape[0]
        n_basins = train_out.max() + 1

        C_range, coeffs, crossval = np.empty((n_C,)), np.empty((n_C, n_basins, n_features)), np.empty((n_C,))

        for idx, data in enumerate(path_data):
            C_range[idx] = data[0]
            coeffs[idx] = data[1]
            crossval[idx] = data[2]

        sort_perm = np.argsort(C_range)
        self._C_range = C_range[sort_perm]
        self._coeffs = coeffs[sort_perm]
        self._crossval = crossval[sort_perm]
        return self._C_range, self._coeffs, self._crossval

    def relevant_features(self, C, normalize_C=True):
        try:
            self._coeffs
            self._C_range
            self._crossval
        except NameError:
            raise ValueError("CV_path not computed.")
        if normalize_C:
            C_range = self._C_range*self._n_samples
        else:
            C_range = self._C_range

        coeffs = self._coeffs
        features = self._features

        C_idx = np.argmin(np.abs(C_range - C))
        selected_coefficients = coeffs[C_idx]

        features_description = dict()
        for state_idx, coef in enumerate(selected_coefficients):
            sparse_coef = csr_matrix(coef)
            coefs_norm = scipy.sparse.linalg.norm(sparse_coef)**2
            model_idxs = sparse_coef.indices
            
            feat_importance = []
            feat_names = []
            feat_val = []
            for feat_idx in model_idxs:
                feat_importance.append(np.around(coef[feat_idx]**2/coefs_norm, decimals=3))
                feat_val.append(coef[feat_idx])
                if self._quadratic_kernel:
                    feat_names.append(decode_quadratic_features(feat_idx, features))
                else:
                    feat_names.append(features[feat_idx])
            sortperm = np.argsort(feat_importance)[::-1]
            data_list = []
            for perm_idx in sortperm:
                #Value,name,importance,index
                data_list.append((feat_val[perm_idx], feat_names[perm_idx], feat_importance[perm_idx], model_idxs[perm_idx]))
            features_description[state_idx] = data_list
        return features_description
    
    def print_relevant_features(self, C, state_names=None, normalize_C=True):
        features_description = self.relevant_features(C, normalize_C=normalize_C)
        n_basins = len(features_description)
        if not state_names:
            state_names = [f'State {idx}' for idx in range(n_basins)]
        
         # padding
        print_lists = []
        for basin_idx in range(n_basins):
            basin_data = features_description[basin_idx]
            print_list = []
            for feat in basin_data:
                print_list.append([f"{np.around(feat[2]*100, decimals=3)}%", f"{feat[1]}"])
            print_lists.append(print_list)
        col_width = max(len(str(row[0])) for row in print_list for print_list in print_lists)
        for basin_idx, print_list in enumerate(print_lists):
            print(f"{state_names[basin_idx]}:")
            for row in print_list:
                print(f"\t {row[0].ljust(col_width)} | {row[1]}")
    
    def plot_relevant_features(self, C, md_object, state_names=None, normalize_C=True, n_configs=100):
        if self._quadratic_kernel:
            raise NotImplementedError()
        
        features_description = self.relevant_features(C, normalize_C=normalize_C)
        n_basins = len(features_description)
        if not state_names:
            state_names = [f'State {idx}' for idx in range(n_basins)]
        
        rows = np.int(np.ceil((n_basins + 1)/3))
        fig = plt.figure(constrained_layout=True, figsize=(9,3*rows))
        gs = GridSpec(rows, 3, figure=fig)

        axes = []
        for basin_idx in range(n_basins):
            axes.append(fig.add_subplot(gs[np.unravel_index(basin_idx, (rows, 3))]))


        for idx in range(n_basins):
            basin_data = features_description[idx]
            ax = axes[idx]
            if len(basin_data) == 1:
                feat_name = basin_data[0][1]
                _data = []
                for basin_idx in range(n_basins):
                    df = md_object._df.loc[ (md_object._df['basin'] == basin_idx) & (md_object._df['selection'] == True)]
                    _data.append(df.filter(regex=feat_name).sample(n=n_configs).values)
                _data = np.asarray(_data)
                feat_bounds = (np.min(_data), np.max(_data))
                for basin_idx in range(n_basins):
                    ax.hist(_data[basin_idx,:], bins=100, range=feat_bounds, alpha=0.5)
                ax.set_xlabel(feat_name)
            elif len(basin_data) >= 2:
                feat_1_name = basin_data[0][1]
                feat_2_name = basin_data[1][1]
                _data_1 = []
                _data_2 = []
                for basin_idx in range(n_basins):
                    df = md_object._df.loc[ (md_object._df['basin'] == basin_idx) & (md_object._df['selection'] == True)]
                    _data_1.append(df.filter(regex=feat_1_name).sample(n=n_configs).values)
                    _data_2.append(df.filter(regex=feat_2_name).sample(n=n_configs).values)
                _data_1 = np.asarray(_data_1)
                _data_2 = np.asarray(_data_2)
                for basin_idx in range(n_basins):
                    ax.scatter(_data_1[basin_idx,:], _data_2[basin_idx,:], alpha=0.5, s=0.5)
                ax.set_xlabel(feat_1_name)
                ax.set_ylabel(feat_2_name)
            else:
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return (fig, axes)

    def plot(self, C, state_names=None, suptitle=None, normalize_C=True):
        try:
            self._coeffs
            self._C_range
            self._crossval
        except NameError:
            raise ValueError("CV_path not computed.")

        coeffs = self._coeffs
        if normalize_C:
            C_range = self._C_range*self._n_samples
        else:
            C_range = self._C_range
        crossval = self._crossval

        n_basins = coeffs.shape[1]

        if not state_names:
            state_names = [f'State {idx}' for idx in range(n_basins)]
        assert len(state_names) == n_basins

        rows = np.int(np.ceil((n_basins + 1)/3))
        fig = plt.figure(constrained_layout=True, figsize=(9,3*rows))
        gs = GridSpec(rows, 3, figure=fig)

        axes = []
        for basin_idx in range(n_basins):
            axes.append(fig.add_subplot(gs[np.unravel_index(basin_idx, (rows, 3))]))
        axes.append(fig.add_subplot(gs[np.unravel_index(n_basins, (rows, 3))]))
        
        C_idx = np.argmin(np.abs(C_range - C))
        if suptitle:
            fig.suptitle(suptitle)
        for idx in range(n_basins):
            ax = axes[idx]
            ax.plot(np.log(1/C_range), coeffs[:,idx,:], 'k-')
            ax.axvline(x = np.log(1/C_range[C_idx]), color='r', linewidth=0.75)
            ax.set_xlim(np.log(1/C_range[-1]), np.log(1/C_range[0]))
            ax.set_xlabel(r"$-\log(C)$")
            ax.set_title(state_names[idx])
            
        ax = axes[-1]
        ax.plot(np.log(1/C_range), crossval, 'k-')
        ax.axvline(x = np.log(1/C_range[C_idx]), color='r', linewidth=0.75)
        ax.set_xlim(np.log(1/C_range[-1]), np.log(1/C_range[0]))
        ax.set_xlabel(r"$-\log(C)$")
        ax.set_title("Score")
        return (fig, axes)
    
    def get_pruned_CVpath(self, C, normalize_C=True):
        try:
            self._coeffs
            self._C_range
            self._crossval
        except NameError:
            raise ValueError("CV_path not computed.")

        assert not self._quadratic_kernel
        if normalize_C:
            C_range = self._C_range*self._n_samples
        else:
            C_range = self._C_range
        C_idx = np.argmin(np.abs(C_range - C))
        selected_coefficients = self._coeffs[C_idx]
        features_mask = prune_idxs(selected_coefficients)
        pruned_dset = (self._dset[0][:,features_mask],self._dset[1][:,features_mask],self._dset[2],self._dset[3])
        pruned_features = self._features[features_mask]
        return CV_path(pruned_dset, pruned_features, quadratic_kernel=True)

def prune_idxs(coeffs):
    extracted_features_idxs = set()
    for coef in coeffs:
        sparse_coef = csr_matrix(coef)
        model_idxs = sparse_coef.indices
        for feat_idx in model_idxs:
            extracted_features_idxs.add(feat_idx)
    extracted_features_idxs = list(extracted_features_idxs)
    extracted_features_idxs = np.asarray(extracted_features_idxs, dtype=np.int_)
    return extracted_features_idxs

def parse_feature_name(name):
    is_quadratic = (name.find('||') >= 0)
    is_squared = False
    if is_quadratic:
        names = name.split(" || ")
        if names[0] == names[1]:
            names = [names[0]]
            is_squared = True
    else:
        names = [name]

    def _is_trig(string, kind):
        is_trig = False
        if (string.find(kind) >= 0):
            is_trig = True
        return is_trig
    
    infos = dict()
    for feat_name in names:
        infos[feat_name] = {
            'sin': _is_trig(feat_name, 'sin_'),
            'cos': _is_trig(feat_name, 'cos_'),
            'squared': is_squared
        }
    return infos
        
