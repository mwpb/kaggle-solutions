# An introduction to machine learning with scikit-learn

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
	* Number of columns = number of features.
* ´.target´: 
	* usually an array specifying the response associated to each sample.

## Learning and predicting

