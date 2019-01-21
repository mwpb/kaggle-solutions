## Supervised learning: predicting an output variable from high-dimensional observations

Supervised learning consists in learning the link between input variables and output variables.
(Estimators associated to supervised learning have a `fit(X, y)` that takes two parameters.)

### Nearest neighbour and the curse of dimensionality

There are various ways of predicting responses in the testing set from responses in the training set.

#### k-nearest neighbours classifier (KNN)

The prediction for a given test point is the response of the closest point in the training data.

The estimator class is loaded as:

```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
```

#### The curse of dimensionality

Sparse data is considered problematic.
One reason for this might be that the relative distances between the points are close to each other and hard to distinguish.

In general the more attributes one has in a dataset the more sparse the data.
This is because the space containing the points increases in dimension.

The curse of dimensionality refers to the increased probability of getting sparse data as one adds attributes.

### Linear model: from regression to sparsity

#### Linear regression

Fits a linear model to the data that minimises the sum of the distances squared.
In scikit-learn it is imported as follows:-

```
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(data_X_train, data_y_train)
```

Then the estimator calculates various values:-
* `regr.coef_`: the coefficients of the linear best fit.
* `regr.predict(data_X_test)`: applies the line of best fit to data.
* `regr.score(data_X_test, data_y_test)`: score measuring how linear the test data is.

#### Shrinkage

Sometimes the attributes that we use are themselves correlated.
In this case the calculation of the least squares solution is more difficult.
(The reason is that at some point we have to invert an almost singular matrix.)
And in fact we get a large variance in the coefficients.

???Unbiased: the mean of the estimator (some kind of combination quantities produced by random variables) is equal to the mean of the random variable itself???

In ridge regression we add a multiple of the identity matrix to the almost singular matrix to make it easier to invert.
(And therefore produce less variance.)
However the new estimator is now no longer unbiased.

In scikit-learn it is imported as follows:-

```
from sklearn import linear_model
regr = linear_model.Ridge(alpha=.1)
```

#### Sparsity

