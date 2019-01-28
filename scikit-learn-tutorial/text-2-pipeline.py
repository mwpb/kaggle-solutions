from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# Training pipeline

categories = [
	'alt.atheism',
	'soc.religion.christian',
	'comp.graphics',
	'sci.med'
]

twenty_train = fetch_20newsgroups(
	subset='train', 
	categories = categories,
	shuffle = True)

print(twenty_train.data)

text_clf = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	# ('clf', MultinomialNB())
	('clf', SGDClassifier())
	])

# text_clf.fit(twenty_train.data, twenty_train.target)

# Testing

twenty_test = fetch_20newsgroups(
	subset='test', 
	categories = categories,
	shuffle = True)

# predictions = text_clf.predict(twenty_test.data)
# print(np.mean(predictions == twenty_test.target))

# Parameter tuning using grid search

pg = {
	'vect__ngram_range': [(1, 1), (1, 2)],
	'tfidf__use_idf': (True, False),
	'clf__alpha': (1e-2, 1e-3)
}

gs_clf = GridSearchCV(text_clf, pg, cv = 5, iid = False, n_jobs = -1)
gs_clf.fit(twenty_train.data[:400], twenty_test.target[:400])
# This seems unreliable!
print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])