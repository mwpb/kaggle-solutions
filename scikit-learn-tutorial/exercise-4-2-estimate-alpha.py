from sklearn import datasets
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils import shuffle
import numpy as np

diabetes = datasets.load_diabetes()
X, y = shuffle(diabetes.data, diabetes.target)
n = len(X)
n_test = n//10
n_train = n - n_test
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]

# alpha using LassoCV - notice that different shuffles give different values!
# in the official solutions they compare the different scores generated 
# by the different folds of LassoCV. 
# Their approach is good because they are sure that all of the data appears at some
# point as testing data.

lasso_cv = LassoCV(cv=3)
lasso_cv.fit(X, y)
print(lasso_cv.alpha_)

# alpha manually using GridSearchCV

pts = np.linspace(0.001, 1, 1000)
pg = [
	{"alpha": pts}
]

lasso = Lasso()
clf = GridSearchCV(estimator = lasso, param_grid = pg, cv = 3, n_jobs = -1)
clf.fit(X, y)
print(clf.best_estimator_.alpha)