import pandas as pd
import numpy as np
from catboost import cv, CatBoost,Pool

import optuna

class CatboostTuner():
    def __init__(self,loss_function='RMSE',eval_metric='RMSE',time_budget=3600,feature_selection=True):
        self.loss_function=loss_function
        self.eval_metric=eval_metric
        self.time_budget=time_budget #for parameter tuning, additional time needed to fit final model
        self.feature_selection=feature_selection
        self.tuned_model=None
        self.final_cv=None
        self.selected_features=None
     
    
    ##tuning plan:
    ##First, tune sequentially
    ##next, remove features by eliminating them on basis of feature importance using optuna
    ##retune within neighborhood of sequential solution
    ##tune learning rate and number of trees
    ##calc final cv statistic
    ##fit final model

    def tune_rsm(self,X,y,w=None,rsm_lb=.1,rsm_ub=1,n_fold=3,learning_rate=.2,subsample=.8,random_strength=1,
        max_depth=4,l2_leaf_reg=3,grow_policy='SymmetricTree',n_trials=10,time_budget=600):
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        pool_data=Pool(data=X,label=y,weight=w)
        def catboost_cv_result(trial):
            params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':subsample,
            'random_strength':random_strength,
            'rsm':trial.suggest_uniform('rsm', rsm_lb,rsm_ub),
            'max_depth':max_depth,
            'grow_policy':grow_policy,
            'l2_leaf_reg':l2_leaf_reg

            }
            cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            #need to generalize this code
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials,timeout=time_budget)
        return([study.best_params['rsm'],study.best_value])

    def tune_depth(self,X,y,w=None,n_fold=3,learning_rate=.2,rsm=.8,subsample=.8,random_strength=1,
        max_depth_lb=2,max_depth_ub=7,l2_leaf_reg=3,grow_policy='SymmetricTree',n_trials=10,time_budget=600):
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        pool_data=Pool(data=X,label=y,weight=w)
        def catboost_cv_result(trial):
            params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':subsample,
            'random_strength':random_strength,
            'rsm':rsm,
            'max_depth':trial.suggest_int('max_depth',max_depth_lb,max_depth_ub),
            'grow_policy':grow_policy,
            'l2_leaf_reg':l2_leaf_reg

            }
            cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            #need to generalize this code
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials,timeout=time_budget)
        return([study.best_params['max_depth'],study.best_value])

    def tune_subsample(self,X,y,w=None,n_fold=3,learning_rate=.2,rsm=.8,subsample_lb=.1,subsample_ub=1,random_strength=1,
        max_depth=4,l2_leaf_reg=3,grow_policy='SymmetricTree',n_trials=10,time_budget=600):
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        pool_data=Pool(data=X,label=y,weight=w)
        def catboost_cv_result(trial):
            params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':trial.suggest_uniform('subsample',subsample_lb,subsample_ub),
            'random_strength':random_strength,
            'rsm':rsm,
            'max_depth':max_depth,
            'grow_policy':grow_policy,
            'l2_leaf_reg':l2_leaf_reg

            }
            cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            #need to generalize this code
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials,timeout=time_budget)
        return([study.best_params['subsample'],study.best_value])

    def tune_regularization(self,X,y,w=None,n_fold=3,learning_rate=.2,rsm=.8,subsample=.8,random_strength_lb=.05,random_strength_ub=3,
        max_depth=4,l2_leaf_reg_lb=.05,l2_leaf_reg_ub=5,grow_policy='SymmetricTree',n_trials=10,time_budget=600):
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        pool_data=Pool(data=X,label=y,weight=w)
        def catboost_cv_result(trial):
            params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':subsample,
            'random_strength':trial.suggest_uniform('random_strength',random_strength_lb,random_strength_ub),
            'rsm':rsm,
            'max_depth':max_depth,
            'grow_policy':grow_policy,
            'l2_leaf_reg':trial.suggest_uniform('l2_leaf_reg',l2_leaf_reg_lb,l2_leaf_reg_ub)

            }
            cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            #need to generalize this code
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials,timeout=time_budget)
        return([study.best_params['random_strength'],study.best_params['l2_leaf_reg'],study.best_value])
    
    def tune_grow_policy(self,X,y,w=None,n_fold=3,learning_rate=.2,rsm=.8,subsample=.8,random_strength=1,
        max_depth=4,l2_leaf_reg=3,n_trials=4,time_budget=600):
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        pool_data=Pool(data=X,label=y,weight=w)
        def catboost_cv_result(trial):
            params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':subsample,
            'random_strength':random_strength,
            'rsm':rsm,
            'max_depth':max_depth,
            'grow_policy':trial.suggest_categorical('grow_policy',['SymmetricTree','Lossguide','Depthwise']),
            'l2_leaf_reg':l2_leaf_reg

            }
            cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            #need to generalize this code
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials,timeout=time_budget)
        return([study.best_params['grow_policy'],study.best_value])

    def tune_iterations(self,X,y,w=None,n_fold=3,learning_rate=.25,rsm=.8,subsample=.8,random_strength=1,       grow_policy="SymmetricTree",
        max_depth=4,l2_leaf_reg=3,time_budget=600):
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        pool_data=Pool(data=X,label=y,weight=w)
        params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':subsample,
            'random_strength':random_strength,
            'rsm':rsm,
            'max_depth':max_depth,
            'grow_policy':grow_policy,
            'l2_leaf_reg':l2_leaf_reg,
            'iterations':4000

            }
        cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)

        min_score=np.min(cv_out["test-RMSE-mean"])
        iterations=np.max(cv_out["iterations"])
        return(iterations,min_score)

    def fine_tune_all(self,X,y,w=None,n_fold=3,learning_rate=.2,rsm=.8,subsample=.8,random_strength=1,grow_policy="SymmetricTree",
        max_depth=4,l2_leaf_reg=3,n_trials=30,time_budget=600):
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        pool_data=Pool(data=X,label=y,weight=w)

        rsm_lb=rsm-.1
        if rsm_lb<=0:
            rsm_lb=.01
        rsm_ub=rsm+.1
        if rsm_ub>1:
            rsm=1
        
        subsample_lb=subsample-.1
        if subsample_lb<=0:
            subsample=.01
        subsample_ub=subsample+.1
        if subsample_ub >1:
            subsample_ub=1
        
        random_strength_lb=random_strength-.5
        random_strength_ub=random_strength+.5
        if random_strength_lb <=0:
            random_strength_lb=.01
        
        max_depth_lb=max_depth-1
        max_depth_ub=max_depth+1

        if max_depth_lb <2:
            max_depth_lb=2
        
        l2_leaf_reg_lb=l2_leaf_reg-.5
        l2_leaf_reg_ub=l2_leaf_reg+.5

        if l2_leaf_reg_lb<=0:
            l2_leaf_reg_lb=0.01
        
        def catboost_cv_result(trial):
            params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':trial.suggest_uniform('subsample',subsample_lb,subsample_ub),
            'random_strength':trial.suggest_uniform('random_strength',random_strength_lb,random_strength_ub),
            'rsm':trial.suggest_uniform('rsm',rsm_lb,rsm_ub),
            'max_depth':trial.suggest_int('max_depth',max_depth_lb,max_depth_ub),
            'grow_policy':grow_policy,
            'l2_leaf_reg':trial.suggest_uniform('l2_leaf_reg',l2_leaf_reg_lb,l2_leaf_reg_ub)

            }
            cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            #need to generalize this code
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)

        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials,timeout=time_budget)
        return([study.best_params,study.best_value])
        
    def tune_feature_selection(self,X,y, model,w=None,n_fold=3,learning_rate=.2,rsm=.8,subsample=.8,random_strength=1,
              grow_policy="SymmetricTree", max_depth=4,l2_leaf_reg=3,n_trials=10,time_budget=600):
        ##use importance scores for feature selection
        early_rounds=round(1/learning_rate+5) #heuristic that seems to work well
        def catboost_cv_result(trial):
            params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':learning_rate,
            'bootstrap_type':'Bernoulli',
            'subsample':subsample,
            'random_strength':random_strength,
            'rsm':rsm,
            'max_depth':max_depth,
            'grow_policy':grow_policy,
            'l2_leaf_reg':l2_leaf_reg

            }

            threshold=trial.suggest_uniform('feat_import_threshold',0,10)
            
            X_subset=self.feature_selection_subsetter(X,model,threshold)
            
            pool_data=Pool(data=X_subset,label=y,weight=w)
            cv_out=cv(params=params,pool=pool_data,nfold=n_fold,early_stopping_rounds=early_rounds,
                    partition_random_seed=2021,verbose=False)
            #need to generalize this code
            out=np.min(cv_out["test-RMSE-mean"])
            return(out)
        study = optuna.create_study(direction='minimize')
        study.optimize(catboost_cv_result, n_trials=n_trials,timeout=time_budget)
        return([study.best_params['feat_import_threshold'],study.best_value])

    def feature_selection_subsetter(self,X,model,threshold):
        imp=model.get_feature_importance()
        var_list=[]
        for i in range(len(imp)):
            if imp[i]> threshold:
                var_list.append(X.columns[i])

        X_out=X[var_list]
        return(X_out)

    def predict(self,X):
        X=X[self.selected_features]
        pred=self.tuned_model.predict(X)
        return(pred)

    def run(self,X,y,w=None,final_learning_rate=.03,tuning_learning_rate=.15,n_fold=3):
        result1=self.tune_rsm(X,y,w,n_fold=n_fold,learning_rate=tuning_learning_rate)
        rsm=result1[0]

        result2= self.tune_grow_policy(X,y,w,rsm=rsm,n_fold=n_fold,learning_rate=tuning_learning_rate)
        grow_policy=result2[0]

        result3= self.tune_depth(X,y,w,rsm=rsm,grow_policy=grow_policy,n_fold=n_fold,learning_rate=tuning_learning_rate)
        max_depth=result3[0]

        result4=self.tune_iterations(X,y,w,learning_rate=final_learning_rate,rsm=rsm,max_depth=max_depth,grow_policy=grow_policy,
            n_fold=n_fold)
        iterations=result4[0]

        params1={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'iterations':iterations,
            'learning_rate':final_learning_rate,
            'bootstrap_type':'Bernoulli',
            'rsm':rsm,
            'grow_policy':grow_policy,
            'max_depth':max_depth,
            'subsample':0.8,
            'random_strength':1,
            'l2_leaf_reg':3
        }
        model1=CatBoost(params=params1)
        model1.fit(X,y,w)

        

        result5=self.tune_feature_selection(X,y,model1,w,rsm=rsm,grow_policy=grow_policy,max_depth=max_depth,n_fold=3,learning_rate=tuning_learning_rate)

        feat_import_threshold=result5[0]

        X=self.feature_selection_subsetter(X,model1,feat_import_threshold)

        #retune first 3

        result1=self.tune_rsm(X,y,w,n_fold=n_fold,learning_rate=tuning_learning_rate)
        rsm=result1[0]

        result2= self.tune_grow_policy(X,y,w,rsm=rsm,n_fold=n_fold,learning_rate=tuning_learning_rate)
        grow_policy=result2[0]

        result3= self.tune_depth(X,y,w,rsm=rsm,grow_policy=grow_policy,n_fold=n_fold,learning_rate=tuning_learning_rate)
        max_depth=result3[0]

        result4=self.tune_subsample(X,y,w,rsm=rsm,max_depth=max_depth,grow_policy=grow_policy,n_fold=n_fold,learning_rate=tuning_learning_rate)
        subsample=result4[0]

        result5=self.tune_regularization(X,y,w,rsm=rsm,max_depth=max_depth,grow_policy=grow_policy,
                subsample=subsample,n_fold=n_fold,learning_rate=tuning_learning_rate)
        random_strength=result5[0]
        l2_leaf_reg=result5[1]

        result6=self.fine_tune_all(X,y,w,rsm=rsm,max_depth=max_depth,grow_policy=grow_policy,
                subsample=subsample,random_strength=random_strength,l2_leaf_reg=l2_leaf_reg,n_fold=n_fold,learning_rate=tuning_learning_rate)

        if result6[1]<result5[1]:
            rsm=result6[0]['rsm']
            max_depth=result6[0]['max_depth']
            subsample=result6[0]['subsample']
            grow_policy=result6[0]['grow_policy']
            random_strength=result6[0]['random_strength']
            l2_leaf_reg=result6[0]['l2_leaf_reg']


        result7=self.tune_iterations(X,y,w,learning_rate=final_learning_rate,rsm=rsm,max_depth=max_depth,grow_policy=grow_policy,
                subsample=subsample,random_strength=random_strength,l2_leaf_reg=l2_leaf_reg,n_fold=n_fold)
        iterations=result7[0]
        self.final_cv=result7[1]

        params={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'iterations':iterations,
            'learning_rate':final_learning_rate,
            'bootstrap_type':'Bernoulli',
            'rsm':rsm,
            'grow_policy':grow_policy,
            'max_depth':max_depth,
            'subsample':subsample,
            'random_strength':random_strength,
            'l2_leaf_reg':l2_leaf_reg
        }
            
        self.selected_features=X.columns
        self.tuned_model=CatBoost(params=params)
        self.tuned_model.fit(X,y,w)
        
    

    





