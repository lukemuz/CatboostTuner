import CatboostTuner
import pandas as pd
from scipy.stats import spearmanr


d_train=pd.read_csv("/Users/lucasmuzynoski/Projects/Numerai2/numerai_datasets/numerai_training_data.csv")

d_val=pd.read_csv("/Users/lucasmuzynoski/Projects/Numerai2/numerai_datasets/numerai_tournament_data.csv")
d_val=d_val[d_val["data_type"]=="validation"]
feature_names = [
        f for f in d_train.columns if f.startswith("feature")
    ]

tuned_catboost=CatboostTuner.CatboostTuner(feature_selection=False,optuna_fine_tune=False)
tuned_catboost.run(X=d_train[feature_names],y=d_train["target"],learning_rate=.03,tuning_learning_rate=.12,nfold=3,cv_type="TimeSeries")

predictions = tuned_catboost.predict(d_val[feature_names])
print("Test MSE:", spearmanr(d_val["target"], predictions))
print(tuned_catboost.params)