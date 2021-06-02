class TuningBudgetController():
    def __init__(self,X,y,w,params,nfold,cv_type,random_seed,time_budget,optuna_fine_tune,feature_selection):
        self.X=X,
        self.y=y
        self.w=w
        self.params=params
        self.nfold=nfold
        self.cv_type=cv_type
        self.random_seed=random_seed
        self.time_budget=time_budget
        self.optuna_fine_tune=optuna_fine_tune
        self.feature_selection=feature_selection
    def evaluate_budget(self):
        ##not yet implemented
        pass
    def get_num_trials(self,param):
        ##currently set to default numbers until timing budget is programmed
        iterations_dict={
            'grow_policy':3,
            'rsm':5,
            'max_depth':4,
            'subsample':5,
            'feature_selection':8,
            'random_strength':5,
            'l2_leaf_reg':5,
            'optuna_fine':10

        }
        out=iterations_dict[param]
        return(out)