# Putting it all together

[website](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html)

## Pipelining

The class 

```
from sklearn.pipeline import Pipeline
```

takes a `steps` list that consists of transformers and a final estimator.
This pipeline can then be passed into the cross-validation classes.
E.g.

```
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
param_grid = {
    'pca__n_components': [5, 20, 30, 40, 50, 64],
    'logistic__alpha': np.logspace(-4, 4, 5),
}
search = GridSearchCV(pipe, param_grid, iid=False, cv=5,
                      return_train_score=False)
```

to perform cross-validation on the entire pipeline.

## Face recognition with eigenfaces

An example.