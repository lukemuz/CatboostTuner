import pandas as pd
import numpy as np

import sklearn
from catboost import cv, CatBoostRegressor, CatBoostClassifier, CatBoost,Pool

import optuna

class AutoMuzzyInterface():
    #A quick and informal interface
  

    #public class methods    
    def fit(self,X,y):
        #returns self
        print ("method to be implemented")
    
    def predict(self,X):
        #returns array
        print ("method to be implemented")
    
    def get_best_model(self):
        #returns model
        print ("method to be implemented")
        
    def get_best_score(self):
        #returns scalar
        print ("method to be implemented")


class BaseEstimatorInterface():
    #A quick and informal interface
    
    #public class methods
    def fit(self,X,y):
        print ("method to be implemented")

    def predict(self,X):
        print ("method to be implemented")
    
    def fit_score_eval(self,X,eval_metric,eval_type):
        print ("method to be implemented")
    
    def get_score_eval(self):
        print ("method to be implemented")

    def set_params(self,params):
        print ("method to be implemented")
    
    def get_params(self):
        print ("method to be implemented")

class CatRegressionEstimator(BaseEstimatorInterface):
    def __init__(self,name,params=None):
        self.model=CatBoost(params)
        self.name=name
        self.score_eval=None

    def fit(self,X,y):
        self.model.fit(X,y)
    def predict(self,X):
        y_pred=self.model.predict(X)
        return(y_pred)
    def fit_score_eval(self,X,y):
        d=Pool(data=X,label=y)
        self.score_eval=cv(pool=d,params=self.params)
    def get_score_eval(self):
        return(self.score_eval)
    def set_params(self,params):
        self.model.set_params(params)
    def get_params(self):
        return(self.model.get_params())



class AutoMuzzy(AutoMuzzyInterface):
    def __init__(self,objective="rmse",eval_metric="rmse"):
        self.objective=objective
        self.eval_metric=eval_metric
        self.model_list=[]
        self.best_model=None
    
    def __initial_catboost_fit(self,X,y):
        ##change default parameters, if desired
        mod=CatRegressionEstimator(name="catboost1")
        #params=whatever
        #mod.set_params(params)
        mod.fit_score_eval(X,y)
        mod.fit(X,y)
        self.model_list.append(mod)
        ##first model is also best, so far
        self.best_model=mod

    def __catboost_naive_tune(self,X,y,n_trials):
        pool_data=Pool(data=X,label=y)
        def catboost_cv_result(trial):
            learning_rate=.25
            early_rounds=round(1/learning_rate+1) #heuristic that seems to work well
            params={
            'loss_function':'RMSE',
            'eval_metric':'RMSE',
            'learning_rate':0.25,
            'bootstrap_type':'Bernoulli',
            'subsample':trial.suggest_uniform('subsample', 0.3, 1),
            'random_strength':trial.suggest_uniform('random_strength',.01,2),
            'rsm':trial.suggest_uniform('rsm', .1,1),
            'max_depth':trial.suggest_categorical('max_depth',[2,3,4,5,6]),
            'grow_policy':trial.suggest_categorical('grow_policy',['SymmetricTree','Depthwise','Lossguide']),
            'l2_leaf_reg':trial.suggest_uniform('l2_leaf_reg',1,5)

            }
            cv_out=cv(params=params,pool=pool_data,nfold=3,shuffle=False,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials)

    def __catboost_sequential_tune(self,X,y,n_trials):
        pool_data=Pool(data=X,label=y)
        def catboost_cv_1(trial):
            learning_rate=.25
            early_rounds=round(1/learning_rate+1) #heuristic that seems to work well
            params={
            'loss_function':'RMSE',
            'eval_metric':'RMSE',
            'learning_rate':0.25,
            'bootstrap_type':'Bernoulli',
            'subsample':.8,
            #'random_strength':trial.suggest_uniform('random_strength',.01,2),
            'rsm':trial.suggest_uniform('rsm', .1,1),
            'max_depth':3,
            'grow_policy':'SymmetricTree'
            #'l2_leaf_reg':trial.suggest_uniform('l2_leaf_reg',1,5)

            }
            cv_out=cv(params=params,pool=pool_data,nfold=3,shuffle=False,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        
        study1 = optuna.create_study(direction='minimize')
        study1.optimize(catboost_cv_1, n_trials=5)
        best_rsm=study1.best_params['rsm']

        def catboost_cv_2(trial):
            learning_rate=.25
            early_rounds=round(1/learning_rate+1) #heuristic that seems to work well
            params={
            'loss_function':'RMSE',
            'eval_metric':'RMSE',
            'learning_rate':0.25,
            'bootstrap_type':'Bernoulli',
            'subsample':.8,
            #'random_strength':trial.suggest_uniform('random_strength',.01,2),
            'rsm':trial.suggest_uniform('rsm', .1,1),
            'max_depth':3,
            'grow_policy':'SymmetricTree'
            #'l2_leaf_reg':trial.suggest_uniform('l2_leaf_reg',1,5)

            }
            cv_out=cv(params=params,pool=pool_data,nfold=3,shuffle=False,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)








#if type in {"regresion","classification"}:
#        self.type=type
#else:
#        print("'type' must be either 'classification' or 'regression'")
