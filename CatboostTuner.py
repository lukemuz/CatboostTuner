
from catboost import CatBoost
from TuningBudgetController import TuningBudgetController
from ParamTuners import FeatureSelectionTuner, IterationsTuner, ParamGridTuner, OptunaFineTuner


class CatboostTuner():
    def __init__(self,loss_function='RMSE',eval_metric='RMSE',time_budget=3600,feature_selection=False,optuna_fine_tune=False,is_minimize=True):
        self.loss_function=loss_function
        self.eval_metric=eval_metric
        self.time_budget=time_budget 
        self.feature_selection=feature_selection
        self.tuned_model=None
        self.final_cv=None
        self.selected_features=None
        self.optuna_fine_tune=optuna_fine_tune
        self.is_minimize=is_minimize

        self.params={
            'loss_function':loss_function,
            'eval_metric':eval_metric,
            'learning_rate':.03,
            'bootstrap_type':'Bernoulli',
            'subsample':.8,
            'random_strength':1,
            'rsm':.8,
            'max_depth':4,
            'grow_policy':'SymmetricTree',
            'l2_leaf_reg':3,
            'iterations':4000

        }
    def _create_rsm_grid(self,num_trials):
        #create functionality to generate search grid based on number of trials 
        # allocated by the time budget controller

        #for now, returning a default grid
        return([.1,.2,.3,.4,.5,.6,.7,.8,.9])

    def _create_depth_grid(self,num_trials):
        #not yet functional
        #returning default grid
        return([3,4,5,6])

    def _create_feature_grid(self,num_trials):
        #not yet functional
        #returning default grid
        return([0,0.001,0.1,0.25,.75,1,3])

    def _create_subsample_grid(self,num_trials):
        #not yet functional
        #returning default grid
        return([.3,.4,.5,.6,.7,.8,.9])

    def _create_random_strength_grid(self,num_trials):
        #not yet functional
        #returning default grid
        return([.5,1,1.5,2])
    
    def _create_l2_leaf_reg_grid(self,num_trials):
        #not yet functional
        #returning default grid
        return([2,2.5,3,3.5,4])

    def predict(self,X):
        if self.feature_selection:
            X_subset=X[self.selected_features]
            pred=self.tuned_model.predict(X_subset)
        else:
            pred=self.tuned_model.predict(X)
        return(pred)

    def run(self,X,y,w=None,learning_rate=0.03,nfold=3,cv_type="Classical",random_seed=2021,tuning_learning_rate=.03):
        ##tuning plan:
        ##First, tune sequentially
        ##next, remove features by eliminating them on basis of feature importance
        ##retune within neighborhood of sequential solution
        ##tune learning rate and number of trees
        ##calc final cv statistic
        ##fit final model

        self.params['learning_rate']=tuning_learning_rate
        #tune grow_policy

        ##get budget
        time_budget=TuningBudgetController(X,y,w,self.params,nfold,cv_type,random_seed,self.time_budget,self.optuna_fine_tune,self.feature_selection)
        
        ##tune grow_policy
        grow_policy_grid=['SymmetricTree','Depthwise','Lossguide']
        grow_policy_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        grow_policy_tuner.tune(X,y,param='grow_policy',param_grid=grow_policy_grid,w=w,nfold=nfold)
        self.params['grow_policy']=grow_policy_tuner.get_best_param()
        
        ##tune tree depth
        max_depth_grid=self._create_depth_grid(num_trials=time_budget.get_num_trials('max_depth'))
        depth_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        depth_tuner.tune(X,y,param='max_depth',param_grid=max_depth_grid,w=w,nfold=nfold)
        self.params['max_depth']=depth_tuner.get_best_param()

        ##tune rsm
        rsm_grid=self._create_rsm_grid(num_trials=time_budget.get_num_trials('rsm'))
        rsm_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        rsm_tuner.tune(X,y,param='rsm',param_grid=rsm_grid,w=w,nfold=nfold)
        self.params['rsm']=rsm_tuner.get_best_param()

        ##
        if self.feature_selection:
            #tune iterations
            iterations_tuner=IterationsTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
            iterations_tuner.tune(X,y,w=w,nfold=nfold)
            params1=self.params
            params1['iterations']=iterations_tuner.get_best_param()

            #fit model to get feature importances
            mod1=CatBoost(params1)
            mod1.fit(X,y=y,sample_weight=w)

            #tune features
            feature_grid=self._create_feature_grid(num_trials=time_budget.get_num_trials('feature_selection'))
            feature_tuner=FeatureSelectionTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
            feature_tuner.tune(X,y,mod1,feature_grid,w=w,nfold=nfold)
            self.selected_features=feature_tuner.get_selected_features(X,mod1)

            #replace X with X_subset
            X=feature_tuner.get_x_subset(X,mod1)

            #retune rsm
            rsm_tuner.tune(X,y,param='rsm',param_grid=rsm_grid,w=w,nfold=nfold)
            self.params['rsm']=rsm_tuner.get_best_param()
        
        #tune subsample
        subsample_grid=self._create_subsample_grid(num_trials=time_budget.get_num_trials('subsample'))
        subsample_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        subsample_tuner.tune(X,y,param='subsample',param_grid=subsample_grid,w=w,nfold=nfold)
        self.params['subsample']=subsample_tuner.get_best_param()

        ##tune random_strength
        rs_grid=self._create_random_strength_grid(num_trials=time_budget.get_num_trials('random_strength'))
        rs_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        rs_tuner.tune(X,y,param='random_strength',param_grid=rs_grid,w=w,nfold=nfold)
        self.params['random_strength']=rs_tuner.get_best_param()

        #tune l2_leaf_reg
        l2_grid=self._create_random_strength_grid(num_trials=time_budget.get_num_trials('l2_leaf_reg'))
        l2_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        l2_tuner.tune(X,y,param='l2_leaf_reg',param_grid=l2_grid,w=w,nfold=nfold)
        self.params['l2_leaf_reg']=l2_tuner.get_best_param()

        if self.optuna_fine_tune:
            optuna_tuner=OptunaFineTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
            optuna_tuner.tune(X,y,w=w,nfold=nfold,time_limit=300)
            optuna_param=optuna_tuner.get_best_param()
            self.params['rsm']=optuna_param['rsm']
            self.params['max_depth']=optuna_param['max_depth']
            self.params['random_strength']=optuna_param['random_strength']
            self.params['l2_leaf_reg']=optuna_param['l2_leaf_reg']
            self.params['subsample']=optuna_param['subsample']

        ##reduce learning rate
        self.params['learning_rate']=learning_rate

        iterations_tuner2=IterationsTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        iterations_tuner2.tune(X,y,w=w,nfold=nfold)
        self.params['iterations']=iterations_tuner2.get_best_param()

        final_mod=CatBoost(self.params)
        final_mod.fit(X,y=y,sample_weight=w)
        self.tuned_model=final_mod







        






        







