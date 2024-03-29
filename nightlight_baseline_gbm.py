# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:14:02 2019

@author: kcsii_000
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score as skl_r2
from sklearn.model_selection import KFold
import utils

mode = 'gbm'

dhs = pd.read_csv('data/dhs_gps.csv', header=0, index_col=0)
nightlight = pd.read_csv('data/nightlight_buckets.csv', 
                         header=0, index_col=0)
#nightlight.set_index('cluster_id', drop=True, inplace=True)

data = dhs.join(nightlight, how='inner')

vacc_cols = ('bcg', 'measles', 'dpt1', 'dpt2', 'dpt3', 
             'polio0', 'polio1', 'polio2', 'polio3',
             'health_card', 'any_vacc')
output = ''
for c in vacc_cols:
#    print(c.center(80, '-'))
    X = data.iloc[:, 13:].to_numpy()
    y = data[c].to_numpy()
    # Get rid of rows with no answers for this response
    X = X[~np.isnan(y)]
    y = y[~np.isnan(y)]
    n = X.shape[0]
    n_train = round(n*.8)
    shuffle = np.random.choice(range(n), size=n, replace=False)
    idx_train = shuffle[:n_train]
    idx_test = shuffle[n_train:]
        
    if mode == 'gbm':
        param_choices = utils.product_dict(
                objective=['regression'],
                num_leaves=(5, 7, 10),
                min_data_in_leaf=(5, 10, 15, 20)
                )
    elif mode == 'linear':
        param_choices = utils.product_dict(
        alpha=np.geomspace(.001, 100, num=6)
        )
    best_params = None
    best_mse = 999999999
    for params in param_choices:
        # Create CV folds
        n_fold = 10
        kf = KFold(n_splits = n_fold)
        pred = np.full_like(y[idx_train], np.nan)
        for train_folds, test_folds in kf.split(X[idx_train]):
            if mode == 'gbm':
                lgb_data = lgb.Dataset(data=X[train_folds], label=y[train_folds])
                gbm = lgb.train(params, lgb_data, 100)
                fold_pred = gbm.predict(X[test_folds])
            elif mode == 'linear':
                model = Ridge(**params, max_iter=1000)
                model.fit(X[train_folds], y[train_folds])
                fold_pred = model.predict(X[test_folds])
            pred[test_folds] = fold_pred
        avg_mse = np.power(y[idx_train]-pred, 2).mean()
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = params
#            print('->', end='')
#        print(str(params) + ': %.3f' % r2)
    
    lgb_data = lgb.Dataset(data=X[idx_train], label=y[idx_train])
    best_gbm = lgb.train(best_params, lgb_data, 100)
    y_pred = best_gbm.predict(X)
    mse = [np.power(y[i]-y_pred[i], 2).mean() for i in (idx_train, idx_test)]
    r2 = [skl_r2(y[i], y_pred[i]) for i in (idx_train, idx_test)]
#    print('train/val/test r2 : %.3f / %.3f / %.3f' % tuple(r2))
#    print('train/val/test mse: %.3f / %.3f / %.3f' % tuple(mse))
    print(c.ljust(12) + ', %.3f, %.3f , %.3f, %.3f' 
          % (r2[0], mse[0], r2[1], mse[1]))
#    print('%.3f'  % r2[2])
    
#
#
#print('lightgbm model'.upper().center(80, '-'))
#print('linear regression model'.upper().center(80, '-'))
#for c in vacc_cols:
#    y = data[c].to_numpy()
#    reg = skl_linreg().fit(X[idx_train], y[idx_train])
#    pred = reg.predict(X[idx_test])
#    mse = (pred-y[idx_test]).mean()
#    r2 = skl_r2(y[idx_test], pred)
#    print(c + ' r2: %.3f, mse: %.3f' % (r2, mse))