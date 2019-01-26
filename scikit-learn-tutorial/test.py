from sklearn import datasets
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph

digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
print(images[0].shape)
connectivity = grid_to_graph(*images[0].shape)