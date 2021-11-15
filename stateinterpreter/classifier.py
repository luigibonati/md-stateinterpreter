import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import concurrent.futures
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from group_lasso import LogisticGroupLasso
from tqdm import tqdm

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

class Classifier():
    def __init__(self, dataset, features):
        self._train_in, self._val_in, self._train_out, self._val_out = dataset
        self._n_samples = self._train_in.shape[0]
        self.features = features
        self._computed = False

    def compute(self, reg, max_iter = 100,  quadratic_kernel=False, groups=None):
        if hasattr(reg, '__iter__') == False:
            reg = np.array([reg])
        _num_reg = len(reg)
        _n_basins = len(np.unique(self._train_out))

        if quadratic_kernel:
            train_in, val_in = quadratic_kernel_featuremap(self._train_in), quadratic_kernel_featuremap(self._val_in)
        else:
            train_in = self._train_in
            val_in = self._val_in
        _n_features = train_in.shape[1]

        if groups is not None:
            if quadratic_kernel:
                assert len(groups) == train_in.shape[1], "Length of group array does not match quadratic features number."
            else:
                assert len(groups) == len(self.features), "Length of group array does not match features number."
            _is_group = True
            def _train_model(idx):
                model = LogisticGroupLasso(groups,group_reg = reg[idx], l1_reg=0, n_iter=max_iter, supress_warning=True, scale_reg='none') 
                #Model Fit
                model.fit(train_in,self._train_out)
                score = model.score(val_in,self._val_out)
                return (idx, model.coef_.T,score, model.classes_)
        else:
            _is_group = False
            def _train_model(idx):
                C = (reg[idx]*self._n_samples)**-1
                model = LogisticRegression(penalty='l1', C=C, solver='saga', multi_class='ovr', fit_intercept=False, max_iter=max_iter) 
                #Model Fit
                model.fit(train_in,self._train_out)
                score = model.score(val_in,self._val_out)
                return (idx, model.coef_,score, model.classes_)

        coeffs =  np.empty((_num_reg, _n_basins, _n_features))
        crossval = np.empty((_num_reg,))
        classes_labels = np.empty((_num_reg, _n_basins), dtype=np.int_)

        if _is_group:
            _raw_data = []
            for reg_idx in tqdm(range(len(reg)), desc='Group Lasso'):
                _raw_data.append(_train_model(reg_idx))
        else:
            _raw_data = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                fut = [executor.submit(_train_model, reg_idx) for reg_idx in range(len(reg))]
                for fut_result in concurrent.futures.as_completed(fut):
                    _raw_data.append(fut_result.result())
                    
        for _datapoint in _raw_data:
            if _is_group:
                idx, coeff, score, classes = _datapoint
            else:
                idx, coeff, score, classes = _datapoint
            coeffs[idx] = coeff
            crossval[idx] = score
            classes_labels[idx] = classes.astype(int)

        
        self._quadratic_kernel=quadratic_kernel 
        self._groups= groups
        self.classes_labels = classes_labels
        self._reg = reg
        self._coeffs = coeffs
        self._crossval = crossval
        if _is_group:
            self._groups_mask = [
                self._groups == u
                for u in np.unique(self._groups)
            ]
        self._computed = True

    def _closest_reg_idx(self, reg):
        assert self._computed, "You have to run Classifier.compute first."
        return np.argmin(np.abs(self._reg - reg))

    def _get_selected(self, reg):
        reg_idx = self._closest_reg_idx(reg)
        coefficients = self._coeffs[reg_idx]
        selected = []
        for coef in coefficients:
            if self._groups is not None:
                coef = np.array([np.linalg.norm(coef[b])**2 for b in self._groups_mask])
            else:
                coef = coef**2
            
            nrm = np.sum(coef)
            if nrm < 1e-10:
                selected.append([])
            else:
                coef = csr_matrix(coef/nrm)
                sort_perm = np.argsort(coef.data)[::-1]
                #idx, weight
                selected.append(list(zip(coef.indices[sort_perm], coef.data[sort_perm])))
        return selected
    
    def print_selected(self, reg, state_names=None):
        selected = self._get_selected(reg)
        reg_idx = self._closest_reg_idx(reg)
        if not state_names:
            state_names = [f'State {idx}' for idx in self.classes_labels[reg_idx]]
        group_names = np.unique(self._groups)

        print_queue = []
        for state in selected:
            _intra_state_queue = []
            for data in state:
                if self._groups is not None:
                    name = group_names[data[0]]
                else: 
                    if self._quadratic_kernel:
                        name = decode_quadratic_features(data[0], self.features)
                    else:
                        name = self.features[data[0]]       
                _intra_state_queue.append([f"{np.around(data[1]*100, decimals=3)}%", f"{name}"])
            print_queue.append(_intra_state_queue)
        
        col_width = 0
        for _intra_state_queue in print_queue:
            for _row in _intra_state_queue:
                if len(str(_row[0])) > col_width:
                    col_width = len(str(_row[0]))
        
        for state_idx, _intra_state_queue in enumerate(print_queue):
            print(f"{state_names[state_idx]}:")
            for row in _intra_state_queue:
                print(f"\t {row[0].ljust(col_width)} | {row[1]}")

    def get_pruned(self, reg):
        selected = self._get_selected(self, reg)
        if self._quadratic_kernel:
            AttributeError("Pruning is not possible on classifiers trained with quadratic kernels.")
        unique_idxs = set()
        for state in selected:
            for data in state:
                unique_idxs.add(data[0])
        if self._groups is not None:
            mask = np.logical_or.reduce([self._groups_mask[idx] for idx in unique_idxs])
        else:
            mask = np.array([False]*len(self.features))
            for idx in unique_idxs:
                mask[idx] = True

        pruned_dset = (self._dset[0][:,mask],self._dset[1][:,mask],self._dset[2],self._dset[3])
        pruned_features = self._features[mask]
        return Classifier(pruned_dset, pruned_features)

    def save(self):
        pass