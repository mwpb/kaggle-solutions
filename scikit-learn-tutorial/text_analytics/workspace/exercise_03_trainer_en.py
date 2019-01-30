import sys, os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

directory = os.path.dirname(os.path.realpath(__file__))
languages_data_folder = directory+'/../data/languages/paragraphs/'
dataset = load_files(languages_data_folder)

docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5)

vectorizer = TfidfVectorizer(
	ngram_range = (1, 3), 
	analyzer = 'char', # The default is to divide on words not characters.
	use_idf = True)

classifier = Perceptron()

clf = Pipeline([
	('vect', vectorizer),
	('classifier', classifier)])

clf.fit(docs_train, y_train)
with open('./trained_en', 'wb') as file:
	pickle.dump((clf, dataset.target_names), file)