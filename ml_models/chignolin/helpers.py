import numpy as np
from scipy.sparse import csr_matrix, data
from scipy.optimize import minimize
from adaptive import Runner, Learner2D #https://github.com/python-adaptive/adaptive
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import concurrent.futures

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False,
    "font.family": 'serif'
})

def load_chignolin_dataset(data_path, regex_filter = 'ca|cphi|sphi|cpsi|spsi|cchi|schi|o-n', sub_sampling = 1, shift=0, shuffle = True, scaling=True):

    folded_path = data_path + '/folded.dat'
    unfolded_path = data_path + '/unfolded.dat'
    headers = pd.read_csv(folded_path, sep=' ',skipinitialspace=True, nrows=0).columns[2:]
    data_paths = [(folded_path, 1), (unfolded_path, -1)]
    raw_inputs = []
    raw_outputs = []
    for data_path in data_paths:
        filename = data_path[0]
        classification_idx = data_path[1] 
        df = pd.read_csv(filename,sep=' ',skipinitialspace=True, header=None,skiprows=1,names=headers,comment='#') 
        #select subset of columns based on names
        filtered_df = df.filter(regex=regex_filter)
        raw_inputs.append(filtered_df.values[shift::sub_sampling, :])
        raw_outputs.append(classification_idx*np.ones(filtered_df.shape[0])[shift::sub_sampling])
    inputs = np.vstack(raw_inputs)
    outputs = np.hstack(raw_outputs)
    
    if shuffle:
        perm = np.random.permutation(outputs.shape[0])
        inputs = inputs[perm]
        outputs = outputs[perm]
    else:
        perm = None
    
    if scaling:
        scaler = StandardScaler(with_mean=True)
        scaler.fit(inputs)
        inputs = (scaler.transform(inputs), scaler.mean_, scaler.var_)
    return inputs, outputs, perm

def get_features(data_path, regex_filter = 'ca|cphi|sphi|cpsi|spsi|cchi|schi|o-n'):
    headers = pd.read_csv(data_path + '/folded.dat', sep=' ',skipinitialspace=True, nrows=0).columns[2:]
    df = pd.read_csv(data_path + '/folded.dat',sep=' ',skipinitialspace=True, header=None,skiprows=1,names=headers,comment='#') 
    features = df.filter(regex=regex_filter).columns
    return features

def enet_path(dataset, C_range, l1_ratio, **kwargs):
    coeffs = []
    train_in, train_out = dataset

    def _train_model(C):
        model = LogisticRegression(penalty='elasticnet', C=C, solver='saga', l1_ratio=l1_ratio, fit_intercept=False, **kwargs) 
        #Model Fit
        model.fit(train_in,train_out)
        return (C, model.coef_)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fut = [executor.submit(_train_model, C) for C in C_range]
        for fut_result in concurrent.futures.as_completed(fut):
            coeffs.append(fut_result.result())

    C_arr = []
    coeff_arr = []
    for data in coeffs:
        C_arr.append(data[0])
        coeff_arr.append(data[1])

    C_arr = np.array(C_arr)
    coeff_arr = np.array(coeff_arr)
    sort_perm = np.argsort(C_arr)

    return C_arr[sort_perm], np.squeeze(coeff_arr[sort_perm])

def reverse_scaling(train_in, mean, var):
    return train_in*var + mean

def train_model(dataset, C, l1_ratio, features, **kwargs):
    model = LogisticRegression(penalty='elasticnet', C=C, solver='saga', l1_ratio=l1_ratio, fit_intercept=False, **kwargs)
    train_in, train_out = dataset
    model.fit(train_in, train_out)
    model.sparse_coef_ = csr_matrix(model.coef_)
    model_idxs = model.sparse_coef_.indices
    weights = []
    names = []
    w_norm = np.linalg.norm(model.sparse_coef_.data)
    for idx in model_idxs:
        coeff = np.around(model.sparse_coef_[0,idx], decimals = 3)
        weight = np.around(coeff**2/w_norm**2, decimals = 2)
        weights.append(weight)
        names.append(features[idx])
    weights = np.array(weights)
    perm_sort = np.argsort(weights)[::-1]
    weights = weights[perm_sort]
    model_idxs = model_idxs[perm_sort]
    names = np.array(names)[perm_sort]
    print("~~~ SELECTED FEATURES ~~~")
    for idx in range(len(weights)):
        print(f"  {idx +1}. {names[idx]} {weights[idx]*100}%")
    return model

def plot_model(trained_model, test_dataset, features):
    test_in, test_out = test_dataset
    accuracy = np.around(trained_model.score(test_in, test_out), decimals = 2)
    model_idxs = trained_model.sparse_coef_.indices
    light_data = np.split(test_in[:,model_idxs], 2) #Assuming already ordered
    
    if len(model_idxs) == 2:
        bdry_sup = np.maximum(np.max(light_data[0][:,0]), np.max(light_data[1][:,0]))
        bdry_inf = np.minimum(np.min(light_data[0][:,0]), np.min(light_data[1][:,0]))
        y_sup = np.maximum(np.max(light_data[0][:,1]), np.max(light_data[1][:,1]))
        y_inf = np.minimum(np.min(light_data[0][:,1]), np.min(light_data[1][:,1]))

        boundary_x = np.linspace(bdry_inf,bdry_sup,2)
        boundary_y = -(boundary_x* trained_model.sparse_coef_[0,model_idxs[0]])/trained_model.sparse_coef_[0,model_idxs[1]]
        #Plotting
        _, ax = plt.subplots()
        ax.set_xlabel(features[model_idxs[0]])
        ax.set_ylabel(features[model_idxs[1]])
        ax.set_xlim(bdry_inf,bdry_sup)
        ax.set_ylim(y_inf,y_sup)
        ax.scatter(light_data[0][:,0], light_data[0][:,1], marker='x', s=1, c='navy', alpha=0.5, label='Folded')
        ax.scatter(light_data[1][:,0], light_data[1][:,1], marker='x', s=1, c='r', alpha=0.5, label='Unfolded')
        ax.set_title(r'Accuracy on $\mathcal{D}_{{\rm test}}$: ' +f'{accuracy}')
        ax.plot(boundary_x, boundary_y, 'k--')

    elif len(model_idxs) ==1:
        _, ax = plt.subplots()
        ax.hist(light_data[0], bins=200, color='navy', alpha=0.5, density=True);
        ax.hist(light_data[1], bins= 200, color='r', alpha=0.5, density=True);
        val = -trained_model.scaled_intercept_/trained_model.scaled_coef_.data[0]
        ax.plot([val,val] ,[0,1], 'k-', lw=0.5)
        ax.text(val, 1.1, 'Decision bound', ha='center')
        ax.set_title(r'Accuracy on $\mathcal{D}_{{\rm test}}$: ' +f'{accuracy}')
        print(f'Decision bound at {val}')

    if len(model_idxs) > 2:
        weights = []
        names = []
        w_norm = np.linalg.norm(trained_model.sparse_coef_.data)
        for idx in model_idxs:
            coeff = np.around(trained_model.sparse_coef_[0,idx], decimals = 3)
            weight = np.around(coeff**2/w_norm**2, decimals = 2)
            weights.append(weight)
            names.append(features[idx])
        weights = np.array(weights)
        perm_sort = np.argsort(weights)[::-1]
        weights = weights[perm_sort]
        model_idxs = model_idxs[perm_sort]
        names = np.array(names)[perm_sort]
        light_data = np.split(test_in[:,model_idxs], 2) #Assuming already ordered

        bdry_sup = np.maximum(np.max(light_data[0][:,0]), np.max(light_data[1][:,0]))
        bdry_inf = np.minimum(np.min(light_data[0][:,0]), np.min(light_data[1][:,0]))
        y_sup = np.maximum(np.max(light_data[0][:,1]), np.max(light_data[1][:,1]))
        y_inf = np.minimum(np.min(light_data[0][:,1]), np.min(light_data[1][:,1]))
        
        #Plotting
        _, ax = plt.subplots(figsize=(7,4))
        ax.set_xlabel(features[model_idxs[0]])
        ax.set_ylabel(features[model_idxs[1]])
        ax.set_xlim(bdry_inf,bdry_sup)
        ax.set_ylim(y_inf,y_sup)
        ax.scatter(light_data[0][:,0], light_data[0][:,1], marker='.', s=0.5, c='navy')
        ax.scatter(light_data[1][:,0], light_data[1][:,1], marker='.', s= 0.5, c='r')
        ax.set_title(r'Accuracy on $\mathcal{D}_{{\rm test}}$: ' +f'{accuracy}')
        ax_feat = ax.inset_axes([1.15, 0.75, 0.25, 0.25])
        
        y_pos = np.arange(len(names))
        
        ax_feat.barh(y_pos, weights, align='center')
        ax_feat.set_yticks(y_pos)
        ax_feat.set_yticklabels(names)
        ax_feat.invert_yaxis()  # labels read top-to-bottom
        ax_feat.set_xlabel('Feature Importance (sum to 1)')
    