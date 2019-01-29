import sys, os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

directory = directory = os.path.dirname(os.path.realpath(__file__))
movie_reviews_data_folder = directory+"/../data/movie_reviews/txt_sentoken/"
dataset = load_files(movie_reviews_data_folder, shuffle=False)

# # split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=None)

# # TASK: Build a vectorizer / classifier pipeline that filters out tokens
# # that are too rare or too frequent

vectorizer = TfidfVectorizer(max_df = 0.8, min_df = 0.2)
classifier = LinearSVC()
pipeline = Pipeline([
    ('vect', vectorizer),
    ('class', classifier)])

# # TASK: Build a grid search to find out whether unigrams or bigrams are
# # more useful.
# # Fit the pipeline on the training set using grid search for the parameters

best_C = 0.1
best_ngram_range = (1, 1)

pg = {
    'vect__ngram_range': [(1, 1), (2, 2)], # best: (1, 1) , (1, 2) is better than both
    'class__C': [1e-4, 1e-3, 1e-2, 1e-1, 1] # best: 0.1
}

gs = GridSearchCV(pipeline, cv = 3, param_grid = pg, n_jobs = -1)
gs.fit(docs_train, y_train)
with open('./trained_review', 'wb') as file:
    pickle.dump(gs, file)