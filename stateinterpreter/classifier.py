import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import concurrent.futures
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
import scipy.sparse.linalg

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
            return (C, model.coef_,score, model.classes_)
        
        path_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut = [executor.submit(_train_model, C) for C in C_range]
            for fut_result in concurrent.futures.as_completed(fut):
                path_data.append(fut_result.result())

        n_features = train_in.shape[1]
        n_C = C_range.shape[0]
        n_basins = len(np.unique(train_out))

        C_range, coeffs, crossval, classes_labels = np.empty((n_C,)), np.empty((n_C, n_basins, n_features)), np.empty((n_C,)), np.empty((n_C, n_basins), dtype=np.int_)

        for idx, data in enumerate(path_data):
            C_range[idx] = data[0]
            coeffs[idx] = data[1]
            crossval[idx] = data[2]
            classes_labels[idx] = data[3].astype(np.int_)


        sort_perm = np.argsort(C_range)

        self._C_range = C_range[sort_perm]
        self._coeffs = coeffs[sort_perm]
        self._crossval = crossval[sort_perm]
        self.classes_labels = classes_labels[sort_perm]

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
            features_description[self.classes_labels[C_idx, state_idx]] = data_list
        return features_description
    
    def unique_features(self,C):
        # get unique features
        relevant_feat = self.relevant_features(C)

        unique_features = set()
        for state in relevant_feat.values():
            for feat in state:
                unique_features.add(feat[1])
        unique_features = list(unique_features)
    
        return unique_features

    def print_relevant_features(self, C, state_names=None, normalize_C=True, file=None):
        features_description = self.relevant_features(C, normalize_C=normalize_C)
        n_basins = len(features_description)
        if not state_names:
            state_names = [f'State {idx}' for idx in features_description.keys()]
        
         # padding
        print_lists = []
        for basin_idx in features_description.keys():
            basin_data = features_description[basin_idx]
            print_list = []
            for feat in basin_data:
                print_list.append([f"{np.around(feat[2]*100, decimals=3)}%", f"{feat[1]}"])
            print_lists.append(print_list)
        col_width = max(len(str(row[0])) for row in print_list for print_list in print_lists)
        for basin_idx, print_list in enumerate(print_lists):
            print(f"{state_names[basin_idx]}:", file=file)
            for row in print_list:
                print(f"\t {row[0].ljust(col_width)} | {row[1]}", file=file)
    
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
        
