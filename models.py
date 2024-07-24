#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:25:00 2022

@author: pedram
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LassoLars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn import ensemble
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSCanonical
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from mlxtend.evaluate import permutation_test
import seaborn as sns
import matplotlib.pyplot as plt
import time

import warnings
warnings.simplefilter("ignore")


def regression_fit(features, y, model, plot_path, feature_labels=[], num_iter=1, max_iter=10000,
                   plot_pred=True, plot_weight=True, save_plot=False, feature_selection=True, selection_method='single', correct_dilution=False,
                   inner_k=10, outer_k=10, feature_type='sub', target='Doors'):
    
    if model == 'Lasso':
        reg_model = Lasso(fit_intercept=True, positive=False, max_iter=max_iter)
        
        p_grid = {"model__alpha": [1e-5,1e-4,1e-3, 1e-2, 0.1, 0.5, 1.0, 2, 5, 10, 1e+2, 1e+3]}
    
    elif model == 'ElasticNet':
        reg_model = ElasticNet(fit_intercept=True, positive=False,max_iter=max_iter)
        
        p_grid = {"model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2, 5, 10.0, 100.0],
                  "model__l1_ratio": np.append(np.linspace(0.1,0.9,num=9),[0.95, 0.99])}
        
        # p_grid = {"model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        #           "model__l1_ratio": [0,5]}
        
    elif model == 'Ridge':
        reg_model = Ridge(fit_intercept=False, positive=False, max_iter=max_iter)
        
        p_grid = {"model__alpha": [1e-5,1e-4,1e-3, 1e-2, 0.1, 1e+2, 1e+3]}
        
    elif model == 'SVR':
        reg_model = SVR(kernel='rbf', max_iter=max_iter)
        
        p_grid = {"model__C": [1e-5,1e-4,1e-3, 1e-2, 0.1, 1e+2, 1e+3, 10000]}
        
    elif model == "DecisionTree":
        reg_model = DecisionTreeRegressor(criterion="squared_error",random_state=0)
        
        p_grid = {"model__max_depth": list(range(1,10)),
                  "model__min_samples_split": [2,3,4,5,6,7,10,20,30,40,50,60]}
        
    elif model == "gboost":
        reg_model = ensemble.GradientBoostingRegressor(n_estimators=500,random_state=42)
        
        # p_grid = {"model__max_depth": list(range(1,10)),
        #           "model__min_samples_split": [2,3,4,5,6,7,10,20,30,40],
        #           "model__learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
        p_grid = {"model__max_depth": list(range(4,5)),
                  "model__min_samples_split": [3,4,5],
                  "model__learning_rate": [0.01, 0.1]}
        
    elif model == "adaboost":
        reg_model = AdaBoostRegressor(n_estimators=500,random_state=42)
        
        p_grid = {"model__learning_rate": [0.1, 0.5, 1, 5]}
        
    elif model == 'PLS':
        reg_model = PLSRegression(max_iter=max_iter)
        
        p_grid = {"model__n_components": [1,2,3,4,5]}
        
    elif model == 'PLSc':
        reg_model = PLSCanonical(max_iter=max_iter, n_components=1)
        
        p_grid = {"model__tol": [1e-5, 1e-6, 1e-7, 1e-8]}
        
    elif model == 'catboost':
        reg_model = CatBoostRegressor(iterations=1000,
                                      depth=8,
                                      loss_function='RMSE',
                                      verbose=False,
                                      random_seed=42)
        
        p_grid = {"model__l2_leaf_reg": list(range(3))}
        
    elif model == 'LARS':
        reg_model = LassoLars(fit_intercept=True, positive=False, max_iter=max_iter)
        
        p_grid = {"model__alpha": [1e-2, 0.1, 0.5, 1.0, 2, 5, 10, 1e+2]}
        
    scale = StandardScaler()
    
    n_features = features.shape[1]
    # n_features = 'auto'
    if feature_selection:
        if selection_method=='single':
            feature_selector = SequentialFeatureSelector(Ridge(), n_features_to_select='auto', tol=0.1, cv=10, direction="forward", scoring='neg_mean_squared_error')


    regressor = Pipeline(steps=[("scaler", scale), ("model", reg_model)])

    scores = []

    coefs = np.zeros([num_iter*outer_k, features.shape[1]])
    feature_imp = {}
    pred_targets = np.array([])
    actual_targets = np.array([])
    sex = np.array([])
    support_all = []
    k = 1

    for i in range(num_iter):
        print('iteration '+str(i+1)+' started')
        outer_cv = KFold(outer_k, shuffle=True, random_state=i)
        inner_cv = KFold(inner_k, shuffle=True, random_state=i)
        
        nested_model = GridSearchCV(estimator=regressor, param_grid=p_grid, cv=inner_cv, scoring='neg_mean_squared_error')

        for train, test in outer_cv.split(features, y):
            
            print('processing fold: ' + str(k))
            start = time.time()

            X_train, X_test = features[train], features[test]
            y_train, y_test = y[:,0][train], y[:,0][test]
            sex_train, sex_test = y[:,1][train].astype(str), y[:,1][test].astype(str)
            sex_test[sex_test=='0.0'] = 'Female'
            sex_test[sex_test=='1.0'] = 'Male'
            
            # feature selection
            if feature_selection: 
                if selection_method=='group':
                    feat_list = np.arange(n_features)
                    np.random.shuffle(feat_list)
                    feat_list = feat_list.tolist()
                    # print('feat_list',feat_list)
                    groups = [[feat_list[i],feat_list[i+1]] for i in range(0,len(feat_list),2)]
                    feature_selector = SFS(Ridge(), k_features='best', floating=0.1, cv=10, forward=True, scoring='neg_mean_squared_error', feature_groups=groups)
                    
                feature_selector.fit(X_train,y_train)
                X_train = feature_selector.transform(X_train)
                print('# selected features: ',X_train.shape[1])
                if selection_method=='group':
                    support = np.full(n_features, False)
                    support[np.array(feature_selector.k_feature_idx_)] = True
                else:
                    support = feature_selector.support_
                
                # print(support)
                support_all.append(support)
            
            nested_model.fit(X_train, y_train)
            
            best_model = nested_model.best_estimator_
            
            best_model.fit(X_train, y_train)
            
            # scores.append(best_model.score(X_test, y_test))
            if feature_selection:
                scores.append(best_model.score(X_test[:,support], y_test))
            else: 
                scores.append(best_model.score(X_test, y_test))
                
            if model == 'PLS' or model == 'PLSc':
                # coefs = np.append(coefs, np.transpose(best_model.named_steps['model'].coef_), axis=0)
                # coefs[f'fold {k}'] = np.transpose(best_model.named_steps['model'].coef_)
                if feature_selection: 
                    coefs[k-1][support==True] = np.squeeze(best_model.named_steps['model'].coef_)
                else:
                    coefs[k-1] = np.squeeze(best_model.named_steps['model'].coef_)
                    
            if model == 'Lasso' or model == 'ElasticNet' or model== 'Ridge':
                # coefs = np.append(coefs, best_model.named_steps['model'].coef_[np.newaxis,:], axis=0)
                # coefs[f'fold {k}'] = best_model.named_steps['model'].coef_[np.newaxis,:]
                if feature_selection: 
                    coefs[k-1][support==True] = best_model.named_steps['model'].coef_[np.newaxis,:]
                else:
                    coefs[k-1] = best_model.named_steps['model'].coef_[np.newaxis,:]
                    
            if model == 'DecisionTree' or model == 'gboost' or model == 'adaboost' or model == 'catboost':
                feature_imp[f'fold {k}'] = best_model.named_steps['model'].feature_importances_[np.newaxis,:]
                # feature_imp = np.append(feature_imp, best_model.named_steps['model'].feature_importances_[np.newaxis,:], axis=0)
            # pred_y = best_model.predict(X_test)
            if feature_selection: 
                pred_y = best_model.predict(X_test[:,support])
            else:
                pred_y = best_model.predict(X_test)
            
            pred_targets = np.append(pred_targets, pred_y)
            actual_targets = np.append(actual_targets, y_test)
            sex = np.append(sex, sex_test)
            
            end = time.time()
            print('fold ' + str(k) + ' took ' + str(round(end-start, 3)) + ' seconds\n')
            k +=1
    
    support_all = np.array(support_all)
    data_dict = {'targets':actual_targets, 'pred_targets':pred_targets, 'sex':sex}
    target_df = pd.DataFrame(data_dict)
    
    corr = np.corrcoef(target_df['pred_targets'],target_df['targets'])[0,1]
    corr_male = np.corrcoef(target_df.loc[target_df['sex']=='Male', 'pred_targets'],target_df.loc[target_df['sex']=='Male', 'targets'])[0,1]
    corr_female = np.corrcoef(target_df.loc[target_df['sex']=='Female', 'pred_targets'],target_df.loc[target_df['sex']=='Female', 'targets'])[0,1]
    
    p_val = permutation_test(target_df['pred_targets'],target_df['targets'],
                               func=lambda x, y: np.abs(np.corrcoef(x, y))[0,1],
                               method='approximate',
                               num_rounds=10000,
                               seed=42)
    if plot_pred:
        x_range = target_df['targets'].max() - target_df['targets'].min()
        y_range = target_df['pred_targets'].max() - target_df['pred_targets'].min()
        sns.jointplot(data=target_df, x="targets", y="pred_targets", hue="sex", space=1, color ='tab:blue', xlim=(target_df['targets'].min()-(x_range*0.05), target_df['targets'].max()+(x_range*0.05)),
                      ylim=(target_df['pred_targets'].min()-(y_range*0.05), target_df['pred_targets'].max()+(y_range*0.05)), joint_kws={'s':100}, ratio=3)
        sns.regplot(x='targets', y='pred_targets',data=target_df, color ='tab:blue', scatter_kws={'alpha':0.5, 's':100, "zorder":-1})
        plt.title('target: '+target+', model: '+model+', features: '+feature_type+'\n'+'r = '+str(round(corr,2))+ ',  p = '+str(round(p_val,4))+'\n'+'r_male = '+str(round(corr_male,2))+',   r_female = '+str(round(corr_female,2)))
        plt.show()
        if save_plot:
            plt.savefig(plot_path+'/prediction_'+model+'_'+target+'_'+feature_type+'.png',dpi=600)
    
    if plot_weight:
        if len(feature_labels) != features.shape[1]:
            feature_labels = [str(x) for x in range(features.shape[1])]
        
        if feature_labels:
            if model == 'DecisionTree' or model == 'gboost' or model == 'adaboost' or model == 'catboost':
                plt.figure('feature importance',figsize=[6,5])
                importances = np.mean(feature_imp,axis=0)
                indices = np.argsort(importances)
                features_name = feature_labels
                plt.title('Feature Importances')
                j = len(feature_labels)# top j importance
                plt.barh(range(j), importances[indices][len(indices)-j:], color='g', align='center')
                plt.yticks(range(j), [features_name[i] for i in indices[len(indices)-j:]])
                plt.xlabel('Relative Importance')
                plt.show()
                if save_plot:
                    plt.savefig(plot_path+'/weights_'+model+'_'+target+'_'+feature_type+'.png',dpi=600) 
            else:
                plt.figure('weights average',figsize=[10,5])
                fig_labels = feature_labels
                coef_means = np.mean(coefs,axis=0)
                plt.bar(fig_labels,coef_means)
                # plt.xlabel("sub region")
                plt.ylabel("weight")
                plt.title('weights average'+'\n'+'target: '+target+', model: '+model+', features: '+feature_type)
                plt.show()
                if save_plot:
                    plt.savefig(plot_path+'/weights_'+model+'_'+target+'_'+feature_type+'.png',dpi=600)

        else:
            plt.figure('weights average',figsize=[20,5])
            fig_labels = ['L_Sub','L_CA1','L_CA2','L_CA3','L_CA4','L_DG','L_SRLM','L_Cyst','R_Sub','R_CA1','R_CA2','R_CA3','R_CA4','R_DG','R_SRLM','R_Cyst']
            coef_means = np.mean(coefs,axis=0)
            plt.bar(fig_labels,coef_means)
            plt.xlabel("sub region")
            plt.ylabel("weight")
            plt.title('weights average'+'\n'+'target: '+target+', model: '+model+', features: '+feature_type)
            plt.show()
            if save_plot:
                plt.savefig(plot_path+'/weights_'+model+'_'+target+'_'+feature_type+'.png',dpi=600)
    
    return target_df, scores, coefs, feature_imp, corr, p_val, best_model, support_all
