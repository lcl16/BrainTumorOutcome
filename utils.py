import numpy as np
from copy import deepcopy
import pandas as pd
from scipy.stats import pearsonr

inset = np.vectorize(lambda x, ref_set: x in ref_set)

def get_top_k_feature(estimator, features, labels, k=None, return_importance=False):
    '''
    TODO: docstring
    '''
    clf = estimator
    clf.fit(train_features, train_labels)
    clf_imp = (clf.feature_importances_)
    clf_feature_rank = np.argsort(-clf_imp)
    
    if return_importance:
        return clf_feature_rank[:k], clf_imp[clf_feature_rank[:k]]
    
    return clf_feature_rank[:k]


def test_clf(dataset, classifier, **kwargs):
    train_dataset, valid_dataset = dataset
    train_features, train_labels = train_dataset
    valid_features, valid_labels = valid_dataset
    
    clf = classifier(**kwargs)
    clf.fit(train_features, train_labels)
    
    predicted_train = clf.predict(train_features)
    train_acc = (predicted_train == train_labels).mean()

    predicted_valid = clf.predict(valid_features)
    valid_acc = (predicted_valid == valid_labels).mean()
    
    return train_acc, valid_acc


def test_xgb(dataset, **kwargs):
    from xgboost.sklearn import XGBClassifier
    return test_clf(dataset, XGBClassifier, **kwargs)

def test_lsvm(dataset, **kwargs):
    from sklearn.svm import LinearSVC
    return test_clf(dataset, LinearSVC, **kwargs)

def data_split(feature, label, fraction=0.8, shuffle=True):
    assert feature.shape[0] == label.shape[0], 'sample number must be exactly same.'
    
    data_size = feature.shape[0]
    train_size = int(data_size * fraction)
    
    if shuffle:
        ind = np.arange(data_size)
        np.random.shuffle(ind)
        feature = feature[ind]
        label = label[ind]
    
    return (feature[:train_size], label[:train_size]), (feature[train_size:], label[train_size:])


def generate_exact_params(params_list):
    exact_list = []
    param_keys = list(params_list.keys())
    if len(param_keys) == 1:
        _param = param_keys[0]
        for item in params_list[_param]:
            exact_list.append({_param:item})
        return exact_list
    else:
        _param = param_keys[0]
        next_set = deepcopy(params_list)
        del next_set[_param]
        current_sets = generate_exact_params(next_set)
        
        for param_value in params_list[_param]:
            _current_sets = deepcopy(current_sets)
            for current_set in _current_sets:
                current_set[_param] = param_value
            exact_list.extend(_current_sets)
        return exact_list

def grid_search(feature, label, eval_func, params_list, replicates = 5, fraction = 0.8):
    whole_list = generate_exact_params(params_list)
    len_whole_list = len(whole_list)
    # TODO: use logging
    print ('Running grid search on %d set of params...' % len_whole_list)
    datasets = []
    baseline = []
    for i in range(replicates):
        datasets.append(data_split(feature, label, fraction=fraction))
        baseline.append((datasets[-1][0][1].mean(), datasets[-1][1][1].mean()))
    
    score_list = []
    for i, params in enumerate(whole_list):
        currnet_param_score_list = []
        for j, dataset in enumerate(datasets):
            print('\rEval %.2f of %d params...      ' % (i+(j+1.0)/replicates, len_whole_list), end='')
            currnet_param_score_list.append(test_xgb(dataset, **params))
        score_list.append(currnet_param_score_list)
        
    return whole_list, score_list, baseline

def pearson(feature, label):
    pearson_list = []
    for i in range(feature.shape[1]):
        pearson_list.append(pearsonr(feature[:,i], label))
    return np.array(pearson_list)

def map_swap(arr):
    result = np.zeros_like(arr)
    for i in arr:
        result[arr[i]] = i
    return result

def select_feature_by_pearson(features, labels, replicates=100, fraction=0.6):
    
    rank_summary = np.zeros(features.shape[1], dtype='float')
    
    for i in range(replicates):
        print('\r %d/%d  ' % (i+1, replicates), end='')
        (train_feature, train_label), _ = data_split(features, labels, fraction=fraction)
        
        train_pearson = pearson(train_feature, train_label)
        
        single_variable_importance = np.argsort(-np.abs(train_pearson[:,0]))
        feature_rank = map_swap(single_variable_importance)
        
        rank_summary += feature_rank

    rank_summary /= replicates
    return rank_summary