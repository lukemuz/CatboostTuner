
import catboost
from TuningBudgetController import TuningBudgetController
from ParamTuners import FeatureSelectionTuner, IterationsTuner, ParamGridTuner


class CatboostTuner():
    def __init__(self,loss_function='RMSE',eval_metric='RMSE',time_budget=3600,feature_selection=True,optuna_fine_tune=True,is_minimize=True):
        self.loss_function=loss_function
        self.eval_metric=eval_metric
        self.time_budget=time_budget #for parameter tuning, additional time needed to fit final model
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
        pass

    def _create_depth_grid(self,num_trials):
        return([3,4,5,6])

    def _create_feature_grid(self,num_trials):
        return([0,0.2,1,3,5])

    def run(self,X,y,w=None,learning_rate=0.03,nfold=3,cv_type="Classical",random_seed=2021,tuning_learning_rate=.15):
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
        #
        grow_policy_grid=['SymmetricTree','Depthwise','Lossguide']
        grow_policy_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        grow_policy_tuner.tune(X,y,param='grow_policy',param_grid=grow_policy_grid,w=w,nfold=nfold)
        self.params['grow_policy']=grow_policy_tuner.get_best_param()
        
        ##tune tree depth
        max_depth_grid=self._create_depth_grid(num_trials=time_budget.get_num_trials('max_depth'))
        depth_tuner=ParamGridTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
        depth_tuner.tune(X,y,param='max_depth',param_grid=max_depth_grid,w=w,nfold=nfold)
        self.params['max_depth']=grow_policy_tuner.get_best_param()

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
            mod1=catboost(params1)
            mod1.fit(X,y=y,w=w)

            #tune features
            feature_grid=self._create_feature_grid(num_trials=time_budget.get_num_trials('feature_selection'))
            feature_tuner=FeatureSelectionTuner(self.params,is_minimize=self.is_minimize,cv_type=cv_type,cv_random_seed=random_seed)
            feature_tuner.tune(X,y,mod1,feature_grid,w=w,nfold=nfold)
            self.selected_features=feature_tuner.get_selected_features

            #replace X with X_subset
            X=feature_tuner.get_x_subset(X,mod1)

            #retune rsm
            rsm_tuner.tune(X,y,param='rsm',param_grid=rsm_grid,w=w,nfold=nfold)
            self.params['rsm']=rsm_tuner.get_best_param()
        
        

        






        







