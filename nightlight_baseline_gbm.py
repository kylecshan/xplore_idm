# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:14:02 2019

@author: kcsii_000
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score as skl_r2
import utils

dhs = pd.read_csv('data/dhs_gps.csv', header=0, index_col=0)
nightlight = pd.read_csv('data/nightlight_buckets_parsed.csv', 
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
    n_train = round(n*.64)
    n_val = round(n*.16)
    shuffle = np.random.choice(range(n), size=n, replace=False)
    idx_train = shuffle[:n_train]
    idx_val = shuffle[n_train:(n_train+n_val)]
    idx_test = shuffle[(n_train+n_val):]
        
    param_choices = utils.product_dict(
            objective=['regression'],
            num_leaves=(3, 5, 7),
            min_data_in_leaf=(5, 10, 15, 20),
            bagging_fraction=(.2, .5, .8),
            bagging_freq=[1]
            )
    best_params = None
    best_r2 = -1
    for params in param_choices:
        lgb_data = lgb.Dataset(data=X[idx_train], label=y[idx_train])
        gbm = lgb.train(params, lgb_data, 100)
        
        pred = gbm.predict(X[idx_val])
        r2 = skl_r2(y[idx_val], pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_params = params
#            print('->', end='')
#        print(str(params) + ': %.3f' % r2)
    
    lgb_data = lgb.Dataset(data=X[idx_train], label=y[idx_train])
    best_gbm = lgb.train(best_params, lgb_data, 100)
    y_pred = best_gbm.predict(X)
    mse = [np.power(y[i]-y_pred[i], 2).mean() for i in (idx_train, idx_val, idx_test)]
    r2 = [skl_r2(y[i], y_pred[i]) for i in (idx_train, idx_val, idx_test)]
#    print('train/val/test r2 : %.3f / %.3f / %.3f' % tuple(r2))
#    print('train/val/test mse: %.3f / %.3f / %.3f' % tuple(mse))
#    print(c.ljust(12) + ', %.3f, %.3f | %.3f, %.3f | %.3f, %.3f' 
#          % (r2[0], mse[0], r2[1], mse[1], r2[2], mse[2]))
    print('%.3f' 
          % r2[2])
    
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