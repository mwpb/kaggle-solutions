"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess whether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
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

    ## Continued in Jupyter as much more convenient than running repeatedly.

    if best_C == None or best_ngram_range == None:
        pg = {
            'vect__ngram_range': [(1, 1), (2, 2)], # best: (1, 1) , (1, 2) is better than both
            'class__C': [1e-4, 1e-3, 1e-2, 1e-1, 1] # best: 0.1
        }

    # # TASK: print the cross-validated scores for the each parameters set
    # # explored by the grid search

        gs = GridSearchCV(pipeline, cv = 3, param_grid = pg, n_jobs = -1)
        gs.fit(docs_train, y_train)
        print(gs.best_params_)

    # # TASK: Predict the outcome on the testing set and store it in a variable
    # # named y_predicted

    vectorizer.set_params({'ngram_range': best_ngram_range})
    classifier.set_params({'C': best_C})
    y_predicted = pipeline.fit(docs_train, y_train)

    # # Print the classification report
    # print(metrics.classification_report(y_test, y_predicted,
    #                                     target_names=dataset.target_names))

    # # Print and plot the confusion matrix
    # cm = metrics.confusion_matrix(y_test, y_predicted)
    # print(cm)

    # # import matplotlib.pyplot as plt
    # # plt.matshow(cm)
    # # plt.show()
