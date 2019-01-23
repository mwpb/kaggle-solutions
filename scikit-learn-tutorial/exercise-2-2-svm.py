from sklearn import datasets, svm, preprocessing
from sklearn.utils import shuffle

iris = datasets.load_iris()
X, y = shuffle(iris.data, iris.target)

# Remove class
y = [c for c in y if (c==1 or c==2)]
X = [X[i] for i, c in enumerate(y) if (c==1 or c==2)]

# Divide into training and testing
n = len(X)
n_test = n//10
n_train = n - n_test
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]

# Normalise variance
X_train = preprocessing.normalize(X_train, norm='l2')

# linear svm

svc = svm.SVC(kernel = "linear")
svc.fit(X_train, y_train)
print(f"linear score: {svc.score(X_test, y_test)}")

# cubic svm

svc = svm.SVC(kernel = "poly", degree = 3, gamma="auto")
svc.fit(X_train, y_train)
print(f"cubic score: {svc.score(X_test, y_test)}")

# rbf svm

svc = svm.SVC(kernel = "rbf", gamma="auto")
svc.fit(X_train, y_train)
print(f"rbf score: {svc.score(X_test, y_test)}")
