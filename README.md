# CatboostTuner
CatboostTuner is a library designed for accurate hyperparameter tuning of Catboost models.  

## Motivation
While the Catboost defaults often give acceptable results, hyperparameter tuning is essential for maximizing the quality of the model.

There are many packages that attempt to deal with this.  Catboost itself has a grid tuner and a random tuner built in.  These work by using a single validation data set to evaluate the choice of hyperparameter.  This works well for some datasets, but for others, this can be too noisy.  In my experience, a better model can be found by using  k-fold cross-validation with k>=3 for evaluation.  

There are also general purpose hyperparameter tuners, most notably Optuna.  Optuna is not specifically designed to work efficiently with Catboost.  In my experience, Optuna works well for a small number of parameters, but requires a very large number of trials when the number of parameters is large.  

In CatboostTuner, hyperparameters are tuned sequentially in the spirit of a "coordinate descent" algorithm.  This reduces the size of the search space and provides a good result relatively quickly (although it has no guarantees of being globally optimal).  Optionally, CatboostTuner uses Optuna to "fine-tune" the result of the sequential tuning.      

Other packages, like AutoGluon and Mljar are fully automatic machine learning model fitting packages.  Given enough time, these can produce excellent results, but in my experience, they can be computational overkill.  By default, they test many very similar model types (e.g. LightGBM, XGBoost and Catboost), when resources could be used to better fine-tune a single model.  

CatboostTuner is designed to be much more lightweight and focused on tuning a single model.  In the future, I may add options for automatic model ensembling, but that is not the current focus.      

CatboostTuner is still under active development, and many features are not yet enabled.  
 

## Installation

CatboostTuner is still being actively developed and has not yet been released to a package manager.  To test the code "as-is", please retrieve code from this repository.  

## Usage Example

```python
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

tuned_catboost=CatboostTuner.CatboostTuner()
tuned_catboost.run(X=X_train,y=y_train,nfold=5)

predictions = tuned_catboost.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, predictions))
print(tuned_catboost.params)
```
## Limitations and Roadmap
Currently, CatboostTuner does not yet budget time effectively.  This is next in development.

CatboostTuner only accepts numeric data at this time.  The Catboost package has a huge advantage of being able to accept categorical data, and this should be leveraged in CatboostTuner.  This is a priority.

The feature_selection option has not produced promising results in some tests.  This logic will be reevaluated.

Usage documentation is needed before being released to a package manager.  

Future considerations:

Automatic Ensembling

Adding other tuning strategies (e.g. Successive Halving)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)