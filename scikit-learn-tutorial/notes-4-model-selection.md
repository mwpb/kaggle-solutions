# Model selection: choosing estimators and their parameters

[website](https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html)

## Score and cross-validated score

The `.score` method of a fitted estimator returns a number between 0 and 1.
The closer to 1 the better the predictions were.

Cross-validation is a technique to determine how well a certain estimator models the data.
In k-fold cross-validation the data is split into k partitions.
As a recipe:-

1. Shuffle the dataset randomly.
2. Split the dataset into k partitions.
3. For each partition train a model on *all the other data* and use this partition as a testing set.
4. We get an array of score of length k.

The function `np.array_split(arr, k)` is useful in this regard...

## Cross-validation generators

In sklearn there is the `KFold` object which exposes a `.split()` method.
Also there is a convenience function `cross_val_score(est, X, y, cv, n_jobs)` which splits and then gets all the scores.
```
from sklearn.model_selection import KFold, cross_val_score
X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
k_fold = KFold(n_splits = 5)
k_fold.split(X) # splits into 5 subsets of equal (or off by one) length
cross_val_score(svc, X, y, cv = k_fold, n_jobs = -1)
```

(Here `n_jobs = -1` is an instruction to use all available CPUs of the machine.)

In fact sklearn has multiple *cross-validation generators*.
(Classes like KFold above.)
Some shuffle the data before splitting.
Others preserve the class distribution within the groups.
There is a list on the webpage.

## Grid-search and cross-validated parameters

We can use cross-validation to help choose the parameters for estimators.

### Grid-search

In sklearn there is a class which automatically chooses parameters from a grid.
The parameter grid is a list of dictionaries. E.g.

```
pg = [
	{'C': [1, 10, 100, 1000], 'kernel':['linear']},
	{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001] ,'kernel':['rbf']},}
]
```

which gives two grids to be explored completely.
By default `GridSearchCV` uses 3-fold cross-validation (in the future perhaps 5-fold).
It stratifies when appropriate.
To use the whole class

```
from sklearn model_selection import GridSearchCV
clf = GridSearchCV(estimator = svc, param_grid = pg, n_jobs = -1)
```

then `clf` exposes the usual estimator methods as well as:-
* `clf.best_score_`
* `clf.best_estimator_.C` for some parameter `C`
* also `cross_val_score(clf, X, y)` gives the actual scores of the grid search.

### Cross-validated estimators

Some estimators have a version that performs cross validation of certain parameters.
Usually these classes have names ending in 'CV' e.g. `LassoCV`.
