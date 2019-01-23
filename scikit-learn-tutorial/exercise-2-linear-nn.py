from sklearn import datasets, neighbors, linear_model
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
n = digits.data.shape[0]
n_test = n//10
n_train = n - n_test
X_train = digits.data[:n_train]
y_train = digits.target[:n_train]
X_test = digits.data[n_train:]
y_test = digits.target[n_train:]

# KNeighbors

kn_est = KNeighborsClassifier()
kn_est.fit(X_train, y_train)
print(f'KNeighbors score: {kn_est.score(X_test, y_test)}')

# Linear regression

lin_est = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1e5, max_iter = 1000)
lin_est.fit(X_train, y_train)
print(f'LogisticRegression score: {lin_est.score(X_test, y_test)}')