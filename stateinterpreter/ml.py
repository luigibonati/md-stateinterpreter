import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from group_lasso import LogisticGroupLasso
from tqdm import tqdm
import warnings
from .utils.plot import plot_regularization_path, plot_classifier_complexity_vs_accuracy
from ._configs import *

__all__ = ["Classifier","prepare_training_dataset"]

def prepare_training_dataset(descriptors, states_labels, n_configs, regex_filter = None, states_subset=None, states_names=None):
    """Sample points from trajectory

    Args:
        n_configs (int): number of points to sample for each metastable state
        regex_filter (str, optional): regex to filter the features. Defaults to '.*'.
        states_subset (list, optional): list of integers corresponding to the metastable states to sample. Defaults to None take all states.
        states_names (list, optional): list of strings corresponding to the name of the states. Defaults to None.

    Returns:
        (configurations, labels), features_names, states_names
    """
    assert len(descriptors) == len(states_labels), "Length mismatch between descriptors and states_labels."
    if regex_filter is not None:
        features = descriptors.filter(regex=regex_filter).columns.values
    else:
        features = descriptors.columns.values
    config_list = []
    labels = []
    
    if isinstance(states_labels, pd.DataFrame):
        pass 
    elif isinstance(states_labels, np.ndarray):
        states_labels = np.squeeze(states_labels)
        columns = ['labels']
        if states_labels.ndim == 2:
            columns.append('selection')
        states_labels = pd.DataFrame(data=states_labels, columns=columns)
    else:
        raise TypeError(
            f"{states_labels}: Accepted types are 'pandas.Dataframe' or 'numpy.ndarray' "
        )
    if not ('selection' in states_labels):
        states_labels['selection'] = np.ones(len(states_labels), dtype=bool)

    states = dict()
    if states_subset is None:
        states_subset = range(len(states_labels['labels'].unique()))

    if states_names is not None:
        assert len(states_names) == len(states_subset), "Length mismatch between states_names and number of unique states."

    for idx, i in enumerate(states_subset):
        if states_names is None:
            states[i] = i
        else:
            states[i] = states_names[idx]

    for label in states_subset:
        #select label
        df = descriptors.loc[ (states_labels['labels'] == label) & (states_labels['selection'] == True)]
        #select descriptors and sample
        replace = False
        if n_configs > len(df):
            warnings.warn("The asked number of samples is higher than the possible unique values. Sampling with replacement")
            replace = True
        if regex_filter is not None:
            config_i = df.filter(regex=regex_filter).sample(n=n_configs, replace=replace).values   
        else:
            config_i = df.sample(n=n_configs, replace=replace).values 
        config_list.append(config_i)
        labels.extend([label]*n_configs)
    labels = np.array(labels, dtype=int)
    configurations = np.vstack(config_list)
    return (configurations, labels), features, states

class Classifier():
    def __init__(self, dataset, features_names, classes_names, rescale=True, test_size=0.25):
        self._X, self._labels = dataset
        self._rescale = rescale
        self._test_size = test_size

        if self._rescale:
            scaler = StandardScaler(with_mean=True)
            scaler.fit(self._X)
            self._X = scaler.transform(self._X)

        self._train_in, self._val_in, self._train_out, self._val_out = train_test_split(self._X, self._labels, test_size=self._test_size)
        
        self._n_samples = self._train_in.shape[0]
        self.features = features_names
        self.classes = classes_names
        self._computed = False

    def compute(self, reg, max_iter = 100,  quadratic_kernel=False, groups=None, warm_start = True):
        if self._computed:
            warnings.warn("Warning: deleting old computed data")
            self._purge()
        if hasattr(reg, '__iter__') == False:
            reg = np.array([reg])
        _num_reg = len(reg)
        _n_basins = len(np.unique(self._train_out))
        self._reg = reg

        if quadratic_kernel:
            train_in, val_in = quadratic_kernel_featuremap(self._train_in), quadratic_kernel_featuremap(self._val_in)
        else:
            train_in = self._train_in
            val_in = self._val_in
        _n_features = train_in.shape[1]

        if groups is not None:
            groups_names, groups = np.unique(groups, return_inverse=True)
            if quadratic_kernel:
                assert len(groups) == train_in.shape[1], "Length of group array does not match quadratic features number."
            else:
                assert len(groups) == len(self.features), "Length of group array does not match features number."
            _is_group = True
            _reg_name = 'estimator__group_reg'
            tmp_model = LogisticGroupLasso(groups, group_reg = reg[0], l1_reg=0, n_iter=max_iter, supress_warning=True, scale_reg='none', warm_start=False) 
            model = OneVsRestClassifier(tmp_model, n_jobs=2)
            
        else:
            _is_group = False
            _reg_name = 'C'
            reg = (reg*self._n_samples)**-1
            model = LogisticRegression(penalty='l1', C=reg[0], solver='liblinear', multi_class='ovr', fit_intercept=False, max_iter=max_iter, warm_start=warm_start)

        coeffs =  np.empty((_num_reg, _n_basins, _n_features))
        crossval = np.empty((_num_reg,))
        _classes_labels = np.empty((_num_reg, _n_basins), dtype=np.int_)

        
        for reg_idx in tqdm(range(len(reg)), desc='Optimizing Lasso Estimator'):
            model.set_params(**{_reg_name: reg[reg_idx]})
            model.fit(train_in,self._train_out)
            crossval[reg_idx] = model.score(val_in,self._val_out)
            _classes_labels[reg_idx] = model.classes_.astype(int)
            if _is_group:
                assert _n_basins == model.classes_.shape[0]
                tmp_coeffs = np.empty((_n_basins, _n_features))
                for est_idx, _e in enumerate(model.estimators_):
                    tmp_coeffs[est_idx] = _e.coef_[:,0]
                coeffs[reg_idx] = tmp_coeffs
            else:
                coeffs[reg_idx] = model.coef_

        self._quadratic_kernel=quadratic_kernel 
        self._coeffs = coeffs
        self._crossval = crossval
        self._classes_labels = _classes_labels
        self._groups = groups
        if _is_group:
            self._groups_names = groups_names
            self._groups_mask = [
                self._groups == u
                for u in np.unique(self._groups)
            ]
        self._computed = True
    
    def _purge(self):
        if __DEV__:
            print("DEV >>> Purging old data")
        if self._computed:
            del self._quadratic_kernel 
            del self._reg
            del self._coeffs
            del self._crossval
            del self._classes_labels
            if self._groups is not None:
                del self._groups_names
                del self._groups_mask
            del self._groups
        self._computed = False

    def _closest_reg_idx(self, reg):
        assert self._computed, "You have to run Classifier.compute first."
        return np.argmin(np.abs(self._reg - reg))

    def _get_selected(self, reg, feature_mode=False):        
        reg_idx = self._closest_reg_idx(reg)
        coefficients = self._coeffs[reg_idx]
        _classes = self._classes_labels[reg_idx]
        selected = dict()
        group_mode = (not feature_mode) and (self._groups is not None)
        for idx, coef in enumerate(coefficients):
            state_name = self.classes[_classes[idx]]
            if group_mode:
                coef = np.array([np.linalg.norm(coef[b])**2 for b in self._groups_mask])
            else:
                coef = coef**2

            nrm = np.sum(coef)
            if nrm < __EPS__:
                selected[state_name] = []
            else:
                coef = csr_matrix(coef/nrm)
                sort_perm = np.argsort(coef.data)[::-1]
                names = []
                for idx in coef.indices:
                    if group_mode:
                        names.append(self._groups_names[idx])
                    else: 
                        if self._quadratic_kernel:
                            names.append(decode_quadratic_features(idx, self.features))
                        else:
                            names.append(self.features[idx])
                #idx, weight, name
                names = np.array(names)
                selected[state_name]= list(zip(coef.indices[sort_perm], coef.data[sort_perm], names[sort_perm]))

            #If only two states the learned models are the same.
            if len(selected.keys()) == 2:
                del_key = selected.keys()[0]
                selected.pop(del_key)
        return selected
    
    def feature_summary(self, reg):
        return self._get_selected(reg, feature_mode=True)

    def print_selected(self, reg):
        selected = self._get_selected(reg)

        print(f"Accuracy: {int(self._crossval[self._closest_reg_idx(reg)]*100)}%")

        for state in selected.keys():
            state_name = 'State ' +  f'{state}' + ':'
            print(state_name)
            for row in selected[state]:
                print(" " + row[2])

    def prune(self, reg, overwrite=False):    
        selected = self._get_selected(reg)
        if self._quadratic_kernel:
            AttributeError("Pruning is not possible on classifiers trained with quadratic kernels.")
        unique_idxs = set()
        for state in selected.values():
            for data in state:
                unique_idxs.add(data[0])
        if self._groups is not None:
            mask = np.logical_or.reduce([self._groups_mask[idx] for idx in unique_idxs])
        else:
            mask = np.array([False]*len(self.features))
            for idx in unique_idxs:
                mask[idx] = True
        
        if overwrite:
            self._train_in = self._train_in[:, mask]
            self._val_in = self._val_in[:, mask]
            self._X = self._X[:, mask]
            self.features = self.features[mask]
            self._purge()
        else:
            X = self._X[:, mask]
            dset = (X, self._labels)
            pruned_features = self.features[mask]
            return Classifier(dset, pruned_features, self.classes, self._rescale, self._test_size)

    def plot_regularization_path(self, reg):
        return plot_regularization_path(self, reg)

    def plot(self):
        return plot_classifier_complexity_vs_accuracy(self)
    
    def save(self, filename):
        pass

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