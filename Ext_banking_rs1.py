#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:45:12 2023

@author: chetan
"""

import numpy as np
import pandas as pd

merge_df = pd.read_csv(r'Data_Ext_Banking_score.csv')


# set seed
seed = 12


## remove non-transactional features

df_raw = merge_df.drop(['business_id', 'lending_business_id'], axis=1)



## train-test split

from model_building import split_test_train, feature_encoding, classification_models
from model_evaluations import model_metrics, feature_importance, probability_bins, cross_validation
# train test split
x_train, y_train, x_test, y_test = split_test_train(df_raw, target_column='target', test_size=0.3, random_state=seed)
print(f'{x_train.shape = }', '|' ,f'{y_train.shape = }', '|' ,f'{x_test.shape = }', '|' ,f'{y_test.shape = }')


# copy to df
df = x_train.copy(deep=True)



# hyperparameters
params_log_reg = {'penalty': 'l2',
                  'random_state': seed,
                  'solver': 'liblinear',
                  'C': 20}

# model fit
logreg_model = classification_models(x_train, y_train, params_log_reg, models=['log_reg'])

# train cv scores
cv_scores = cross_validation(logreg_model, x_train, y_train, scoring='roc_auc', folds=3, seed=seed)
print('CV Scores -',np.round(cv_scores, 2))
print('Mean of CV Scores -',np.round(np.mean(cv_scores),2))

# train score
# model_metrics(logreg_model.predict(transformed_vars[feat_list]), np.array(y_train), logreg_model.predict_proba(transformed_vars[feat_list]))


# Feature importance
feat_imp = feature_importance(logreg_model, x_train, show_plot=True)

feat_imp.sort_values(by='importance', ascending=False)

# coeff
coeff1 = pd.DataFrame(zip(x_train.columns, np.transpose(logreg_model.coef_.tolist()[0])), columns=['features', 'coef'])

logreg_model.coef_
logreg_model.intercept_


# test cv scores
cv_scores = cross_validation(logreg_model, x_test, y_test, scoring='roc_auc', folds=3, seed=seed)
print('CV Scores -',np.round(cv_scores, 2))
print('Mean of CV Scores -',np.round(np.mean(cv_scores),2))


# test score
# model_metrics(logreg_model.predict(X_test[feat_list]), np.array(y_test), logreg_model.predict_proba(X_test[feat_list]))



## Model Evaluation - KS & ROC AUC

def ks(target=None, prob=None):
    data = pd.DataFrame()
    data['y'] = target
    data['y'] = data['y'].astype(float)
    data['p'] = prob
    data['y0'] = 1- data['y']
    data['bucket'] = pd.qcut(data['p'].rank(method='first'), 5)
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['p']
    kstable['max_prob'] = grouped.max()['p']
    kstable['events'] = grouped.sum()['y']
    kstable['nonevents'] = grouped.sum()['y0']
    kstable = kstable.sort_values(by='min_prob', ascending=False).reset_index(drop=True)
    kstable['event_rate'] = (kstable.events / data['y'].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable['nonevents'] /  data['y0'].sum()).apply('{0:2%}'.format)
    kstable['cum_eventrate'] = (kstable.events / data['y'].sum()).cumsum()
    kstable['cum_noneventrate'] = (kstable.nonevents / data['y0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100
    kstable['bad_rate'] = (kstable['events'] / (kstable['events'] + kstable['nonevents'])) * 100
    
    # formatting
    kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,6)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    print(kstable)
    
    # Display KS
    print("KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return kstable


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def ml_modelling(model, x_train, x_test, y_train, y_test):
    # eval_set = [(X_test, y_test)]
    model.fit(x_train, y_train)
    
    # threshold tuning
    y_hat = model.predict_proba(x_test)
    y_hat = y_hat[:,1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    # predictions
    y_pred1 = model.predict_proba(x_train)
    y_pred2 = model.predict_proba(x_test)
    
    return model, y_pred1, y_pred2, best_thresh

logreg_model, y_hat1, y_hat2, thresh = ml_modelling(logreg_model, x_train, x_test, y_train, y_test)


def evaluate_model(y_actual, y_pred, y_hat):
    y_actual = y_actual.astype(y_pred.dtype)
    print(classification_report(y_actual, y_pred))
    print(confusion_matrix(y_actual, y_pred))
    print(roc_auc_score(y_actual, y_hat))

y_pred1 = y_hat1[:,1] >= thresh
y_pred2 = y_hat2[:,1] >= thresh

evaluate_model(y_train, y_pred1, y_hat1[:,1])
evaluate_model(y_test, y_pred2, y_hat2[:,1])

train_ks_thresh = ks(y_train, y_hat1[:,1])
test_ks_thresh= ks(y_test, y_hat2[:,1])




#
### Credit scoring part

merge_df['pred_proba'] = logreg_model.predict_proba(merge_df[x_train.columns.tolist()])[:,1]

merge_df['odds'] = merge_df['pred_proba'] / (1-merge_df['pred_proba'])

merge_df['log_odds'] = np.log(merge_df['odds'])

merge_df['Ext_banking_rs1'] = np.round(589.3 - (72.1 * merge_df['log_odds']))


#bin
col         = 'Ext_banking_rs1'
conditions  = [ merge_df[col] <= 647, 
               (merge_df[col] > 647) & (merge_df[col]<= 678),
               (merge_df[col] > 678) & (merge_df[col]<= 713),
               (merge_df[col] > 713) & (merge_df[col]<= 756),
                merge_df[col] > 756 ]

choices     = [1, 2, 3, 4, 5]
    
merge_df["Ext_banking_rs1_bin"] = np.select(conditions, choices, default=np.nan)
















