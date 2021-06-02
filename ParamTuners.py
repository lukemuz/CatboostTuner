import pandas as pd
import numpy as np
from catboost import cv, CatBoost,Pool

import optuna

class CVHandler():
    def __init__(self,params,is_minimize=True,nfold=3,cv_type="Classical",random_seed=2021):
        self.params=params
        self.is_minimize=is_minimize
        self.nfold=nfold
        self.cv_type=cv_type
        self.early_rounds=round(1/params['learning_rate']+5)
        self.random_seed=random_seed
        self.cv_out=None

    def run_cv(self,pool_data):
        cv_out=cv(params=self.params,pool=pool_data,nfold=self.nfold,early_stopping_rounds=self.early_rounds,
                    partition_random_seed=self.random_seed,verbose=False)
        self.cv_out=cv_out
    
    def get_best_score(self):
        eval_metric=self.params['eval_metric']
        eval_name='test-'+eval_metric+'-mean'
        if self.is_minimize:
            best_score=np.min(self.cv_out[eval_name])
        else:
            best_score=np.max(self.cv_out[eval_name])
        return(best_score)
    def get_iterations(self):
        return(np.max(self.cv_out['iterations']))

class ParamTuner():
    def __init__(self,params,
        is_minimize=True,cv_type="Classical",cv_random_seed=2021):
        '''self.rsm=rsm
        self.learning_rate=learning_rate
        self.subsample=subsample
        self.random_strength=random_strength
        self.max_depth=max_depth
        self.l2_leaf_reg=l2_leaf_reg
        self.grow_policy=grow_policy
        self.loss_function=loss_function
        self.eval_metric=eval_metric'''

        self.is_minimize=is_minimize
        self.cv_type=cv_type
        self.cv_random_seed=cv_random_seed
        self.params=params
        self.result={}
        self.is_tuned=False

    def tune(self,X,y,param_lb=None,param_ub=None,w=None,nfold=3,time_limit=300):
        pass

    def cv_result(self,pool_data,params,nfold=3,is_minimize=True,cv_type="Classical",random_seed=2021):
        
        cv_handler=CVHandler(params,nfold=nfold,is_minimize=is_minimize,cv_type=cv_type,random_seed=random_seed)
        cv_handler.run_cv(pool_data)
        out=cv_handler.get_best_score()
       
        return(out)
    
    def get_best_param(self):
        pass
        
    

class ParamGridTuner(ParamTuner):

    def tune(self,X,y,param,param_grid,w=None,nfold=3,time_limit=300):
        pool_data=Pool(data=X,label=y,weight=w)
        param_dist=self.params
        result_dict={}
        for param_iteration in param_grid:
            param_dist[param]=param_iteration
            cv_out=self.cv_result(pool_data,param_dist,nfold=nfold,is_minimize=self.is_minimize,cv_type=self.cv_type,random_seed=self.cv_random_seed)
            result_dict[param_iteration]=cv_out
        self.result=result_dict
        self.is_tuned=True
    
    def get_best_param(self):
        first_bool=True
        for param_iter in self.result:
            if first_bool == True:
                min_result=self.result[param_iter]
                best_param=param_iter
                first_bool=False
            else:
                if self.result[param_iter]<min_result:
                    min_result=self.result[param_iter]
                    best_param=param_iter
            
        return(best_param)

    
class OptunaFineTuner(ParamTuner):
    def tune(self,X,y,w=None,nfold=3,time_limit=300):
        pool_data=Pool(data=X,label=y,weight=w)
        param_init=self.params
        rsm_lb=param_init['rsm']-.1
        if rsm_lb<=0:
            rsm_lb=.01
        rsm_ub=param_init['rsm']+.1
        if rsm_ub>1:
            rsm=1
        
        subsample_lb=param_init['subsample']-.1
        if subsample_lb<=0:
            subsample=.01
        subsample_ub=subsample+.1
        if subsample_ub >1:
            subsample_ub=1
        
        random_strength_lb=param_init['random_strength']-.5
        random_strength_ub=param_init['random_strength']+.5
        if random_strength_lb <=0:
            random_strength_lb=.01
        
        max_depth_lb=param_init['max_depth']-1
        max_depth_ub=param_init['max_depth']+1

        if max_depth_lb <2:
            max_depth_lb=2
        
        l2_leaf_reg_lb=param_init['l2_leaf_reg']-.5
        l2_leaf_reg_ub=param_init['l2_leaf_reg']+.5

        if l2_leaf_reg_lb<=0:
            l2_leaf_reg_lb=0.01

        def optuna_cv(trial):
            param_trial={
            'loss_function':self.loss_function,
            'eval_metric':self.eval_metric,
            'learning_rate':param_init['learning_rate'],
            'bootstrap_type':'Bernoulli',
            'subsample':trial.suggest_uniform('subsample',subsample_lb,subsample_ub),
            'random_strength':trial.suggest_uniform('random_strength',random_strength_lb,random_strength_ub),
            'rsm':trial.suggest_uniform('rsm',rsm_lb,rsm_ub),
            'max_depth':trial.suggest_int('max_depth',max_depth_lb,max_depth_ub),
            'grow_policy':param_init['grow_policy'],
            'l2_leaf_reg':trial.suggest_uniform('l2_leaf_reg',l2_leaf_reg_lb,l2_leaf_reg_ub)

            }
            result=self.cv_result(pool_data,param_trial,nfold=nfold,is_minimize=self.is_minimize,cv_type=self.cv_type,random_seed=self.cv_random_seed)
            return(result)

        study = optuna.create_study(direction='minimize')
        study.optimize(optuna_cv, n_trials=100,timeout=time_limit)
        self.result=study.best_params
        self.is_tuned=True
        
    def get_best_param(self):
        return self.result

class FeatureSelectionTuner(ParamTuner):
    def feature_selection_subsetter(self,X,model,threshold):
        imp=model.get_feature_importance()
        var_list=[]
        
        for i in range(len(imp)):
            if imp[i]> threshold:
                var_list.append(X.columns[i])
        X_out=X[var_list]
        return(X_out)

    def tune(self,X,y,model,threshold_grid,w=None,nfold=3,time_limit=300):
        result_dict={}
        for threshold in threshold_grid:
            X_subset=self.feature_selection_subsetter(X,model,threshold)
            pool_data=Pool(data=X_subset,label=y,weight=w)
            cv_out=cv_out=self.cv_result(pool_data,self.params,nfold=nfold,is_minimize=self.is_minimize,cv_type=self.cv_type,random_seed=self.cv_random_seed)
            result_dict[threshold]=cv_out
        self.result=result_dict
        self.is_tuned=True
        

    def get_best_param(self):
        first_bool=True
        for param_iter in self.result:
            if first_bool == True:
                min_result=self.result[param_iter]
                best_param=param_iter
                first_bool=False
            else:
                if self.result[param_iter]<min_result:
                    min_result=self.result[param_iter]
                    best_param=param_iter
            
        return(best_param)
    
    def get_x_subset(self,X,model):
        if self.is_tuned:
            threshold=self.get_best_param()
            return(self.feature_selection_subsetter(X,model,threshold))
        else:
            print("Object not tuned")
    
    def get_selected_features(self,X,model):
        if self.is_tuned:
            threshold=self.get_best_param()
            imp=model.get_feature_importance()
            var_list=[]
            for i in range(len(imp)):
                if imp[i]> threshold:
                    var_list.append(X.columns[i])
            
            return(var_list)
        else:
            print("Object not tuned")
        

class IterationsTuner(ParamTuner):

    def tune(self,X,y,w=None,nfold=3,time_limit=300):
        pool_data=Pool(data=X,label=y,weight=w)
        cv_handler=CVHandler(self.params,nfold=nfold,is_minimize=self.is_minimize,cv_type=self.cv_type,random_seed=self.cv_random_seed)
        cv_handler.run_cv(pool_data)
        self.result=cv_handler
        self.is_tuned=True


    def get_best_param(self):
        return(self.result.get_iterations())



        

        

        
        
        

