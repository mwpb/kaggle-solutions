# Supervised learning: predicting an output variable from high-dimensional observations

[website](https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)

Supervised learning consists in learning the link between input variables and output variables.
(Estimators associated to supervised learning have a `fit(X, y)` that takes two parameters.)

## Nearest neighbour and the curse of dimensionality

There are various ways of predicting responses in the testing set from responses in the training set.

### k-nearest neighbours classifier (KNN)

The prediction for a given test point is the response of the closest point in the training data.

The estimator class is loaded as:

```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
```

### The curse of dimensionality

Sparse data is considered problematic.
One reason for this might be that the relative distances between the points are close to each other and hard to distinguish.

In general the more attributes one has in a dataset the more sparse the data.
This is because the space containing the points increases in dimension.

The curse of dimensionality refers to the increased probability of getting sparse data as one adds attributes.

## Linear model: from regression to sparsity

### Linear regression

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

### Shrinkage

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

This has the effect of decreasing the effect of some attributes that are correlated to other attributes.
(We can interpret ridge regression by solving the minimisation problem of least squares subject to an additional constraint.
This constraint is that the vector of regression coefficients has magnitude less that a given value \alpha chosen in advance.)

### Sparsity (and LASSO)

An approach called the LASSO (least absolute shrinkage and selection operator) is used to actually eliminate some attributes.
This reduces dimensionality.

In a similar way to ridge regression it minimises the least square distance and adds an additional constraint.
In LASSO we insist that the sum of the absolute values of the regression coefficients is less than a given value \alpha chosen in advance.
In order to satisfy this l^1 constraint the regression coefficients are constrained to a 'diamond' around the origin.
So when performing the minimisation (imagine a line moving towards the diamond) we are very likely to set one variable to zero.

In sckit-learn we use `regr = linear_model.Lasso()` to construct the estimator and `regr.setParams(alpha = a)` to specify the minimisation problem.

### Classification

We can apply the above methods to a binary classification also.
(Recall that we can model multi-class classification using several binary classification in several ways.)
Instead of using linear regression we first pass the data through a sigmoid function.
This function redistributes so that the data is much more likely to be closer to 0 or 1.
(And also rules out value less than 0 and greater than 1.)

In sklearn we use `linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')` as the estimator.

## Support vector machines

### Linear SVMs

A linear SVM finds a line (hyper-plane...) that separates the data into two classes.
It maximises the margin between the plane and either of the two datasets.
As such it is a (quadratic?) optimisation problem.
(This estimator takes a parameter `c`.
The higher the parameter the less mistaken classifications are tolerated.)

SVMs works better on data whose standard deviation is normalised (i.e. equals 1).
In sklearn there is a function for this:

```
from sklearn import preprocessing
preprocessing.normalize(X, norm='l2')
```

As such SVMs can be used for either:
* binary classification (SVC): either side of the line
* regression (SVR): use this as a 'line of best fit'

In sklearn an SVC estimator is constructed as follows:
```
from sklearn import svm
svc = svm.SVC(kernel = "linear", C = c)
```

### Using kernels

When a decision function does not suffice it is possible to use:-
* a polynomial separator: `svc = svm.SVC(kernel = "poly", degree = 3)`
* a radial basis function: `svc = svm.SVC(kernel = "rbf")`

(It seems that the radial basis function is applied pairwise.)