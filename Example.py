import CatboostTuner
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

housing = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(housing.data, columns=housing.feature_names),
    housing.target,
    test_size=0.25,
    random_state=123,
)

tuned_catboost=CatboostTuner.CatboostTuner(feature_selection=False,optuna_fine_tune=False)
tuned_catboost.run(X=X_train,y=y_train,learning_rate=.03,tuning_learning_rate=.03,nfold=5)

predictions = tuned_catboost.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, predictions))
print(tuned_catboost.params)