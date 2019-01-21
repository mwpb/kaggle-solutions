# A tutorial on statistical learning for scientific data processing

Statistical learning aims to draw conclusions about the data.

## Statistical learning: the setting and the estimator object in scikit-learn

### Datasets

Given a dataset `dset`:
* `dset.shape`: returns the number of samples (rows) and the number of features (columns)

If the return value isn't a pair then we should flatten it into a pair.
To do this we use the `reshape` command.

### Estimator objects

In scikit-learn an estimator is any object that learns from data.

* `estimator = Estimator(param1 = 1, param2 = 2)`: construction.
* `estimator.param1`: hyper-parameters are accessible directly.
* `estimator.estimated_param_`: parameters calculated by the estimator after fitting are accessible with an underscore.
* `estimator.fit(data)`: trains the estimator on `data`.