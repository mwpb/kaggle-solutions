# An introduction to machine learning with scikit-learn

[webpage](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

## Machine learning: the problem setting

* Given some known data (training set) we try to predict unknown data (testing set).

Learning problems fall into two broad categories:-

1. Supervised learning: there are specified response variables that we want to predict.
	* Classification: when the response variable is discrete.
	* Regression: when the response variable is continuous.
2. Unsupervised learning: no specified response variables. The goal could be:-
	* Clustering: to discover groups of similar examples.
	* Density estimation: to determine the distribution of data.
	* Visualisation: to project the data down to two or three dimensions to plot.

## Scikit-learn datasets

A dataset is a wrapper around data that provides metadata.
This is done using a dictionary.
Common attributes include:-
* ´.data´: a 2D array that constitutes the data of the input variables.
	* Number of rows = number of samples.
	* Number of columns = number of features/attributes.
* ´.target´: 
	* usually an array specifying the response associated to each sample.

## Learning and predicting

We *fit* an *estimator* using the previous responses to *predict* the future responses.

We create an estimator that for the moment will be a black box.

```
from sklearn import svm
bbox = svm.SVC(gamma=0.001, C=100.)
```

The arguments to `SVC` are called hyper-parameters.
Finding good values for these requires extra work.

For the fitting process we pass in the training data and responses.

```
bbox.fit(digits.data[:no_training_samples], digits.target[:no_training_samples])
```

For the prediction step we pass in the testing data.

```
bbox.predict(digits.data[no_training_samples:])
```

## Model persistence

* Built in `pickle` is possible but perhaps inefficient on large datasets.
* The [joblib](https://joblib.readthedocs.io/en/latest/) module is good for long-term persistence and large datasets. (Only pickles to a file not a string.)
* The [feather](https://github.com/wesm/feather) module is good for short-term storage and large datasets.

The standard caveats apply about de-serialising untrusted data for all methods.

## Conventions

### Type casting

Unless otherwise specified input will be cast to `float64`.

In regression the continuous response variables are cast to `float64`.
In classification the discrete categories are not altered. 
(So if one uses an integer representation one gets integers etc..)

### Refitting and updating parameters

Calling `.fit()` more than once will overwrite what was learned previously.

