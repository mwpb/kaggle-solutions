# Unsupervised learning: seeking representations of the data

[website](https://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html)

## Clustering: grouping observations together

Aim: split the data into well-separated partitions called clusters.

As an application, after clustering we can compress data by replacing each observation with the man of its cluster.
(The website gives a 'posterisation' example.)

### K-means clustering

Finds k clusters defined by the property:
each observation falls into the cluster whose means is closest.
(This is in general a hard problem as one considers all clusters.
However there are heuristic algorithms that converge to a local extreme point.)

In sklearn we use:
```
from sklearn import cluster
k_means = cluster.KMeans(n_clusters = 3)
k_means.fit(X)
```

### Hierarchical clustering

In general these types of clustering fall into one of two categories:-
* agglomerative clustering (bottom-up):
	* each observation starts in its own cluster
	* then clusters are merged to minimise a 'linkage' condition
* divisive clustering (top-down):
	* all observations start in the same cluster
	* iteratively splits into more clusters
which means that top-down is better for fewer clusters and vice-versa.

#### Connectivity constrained clustering

Given an image we define:-
* what it means for two pixels to be similar
* what is means for two pixels to be neighbouring
and then create equivalence classes generated by identifying similar neighbours.

In sklearn the similarity is handled by the clustering class being used.
In addition we can add connectivity constraints to prevent undesirable clustering.
(It may be undesirable to cluster far away points on an image for example.)

```
from sklearn.feature_extraction.image import grid_to_graph
cty = grid_to_graph(x_dim, y_dim, ?z_dim)
```

and then this is passed into a clustering class for instance:

```
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 5, connectivity = cty)
```

to constrain the clustering algorithm.

#### Feature agglomeration

It is also possible to perform clustering on the features.
This is useful if one wants to avoid the 'curse of dimensionality'.

In sklearn the interface is similar:
```
from sklearn.cluster import FeatureAgglomeration
agglo = cluster.FeatureAgglomeration(n_clusters=32, connectivity = cty)
```

## Decompositions: from signal to components and loading

### Principal component analysis: PCA

Principal component analysis selects an orthogonal basis iteratively:-
* with respect to the first basis vector the data has the largest possible variance
* the next basis vector has the largest possible variance subject to the condition that it is orthogonal to all the previous vectors in the basis.

In sklearn:

```
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
```
and then we have attributes:-
* `pca.components_`: shows the new orthogonal basis
* `pca.explained_variance_`: the variance of the data in the directions of the new basis

### Independent component analysis: ICA

Attempts to split a set of observations into a sum of independent components.
A heuristic conclusion from the central limit theorem:
the sum of independent non-Gaussian random variables is more Gaussian than each.
The ICA simultaneously finds the components which together are most independent.

* The following is called projection pursuit: the algorithm returns a basis for which the first vector is the vector in which the distribution is the most non-Gaussian. According to wikipedia this is not ICA.

In sklearn we use:
```
from sklearn import decomposition
ica = decomposition.FastICA(n_component = None)
```

and then we have attributes:-
* `ica_components_`: the unmixing matrix taking the data to the independent components
* `ica.mixing_`: the mixing matrix that takes the components to the original data