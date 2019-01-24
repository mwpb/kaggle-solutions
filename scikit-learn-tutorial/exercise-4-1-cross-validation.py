import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn import datasets, svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel = "linear")
C_s = range(-10, 0)
log_C_s = np.logspace(-10, 0, 10)

k = 5
k_fold = KFold(n_splits = 5)

c_vals = []
score_vals = []

for i, c in enumerate(C_s):
	svc.set_params(C = log_C_s[i])
	scores = cross_val_score(svc, X, y, cv = k_fold, n_jobs = -1)
	for score in scores:
		score_vals.append(score)
		c_vals.append(c)	

print(c_vals)
print(score_vals)
plt.plot(c_vals, score_vals, 'ro')
plt.show()