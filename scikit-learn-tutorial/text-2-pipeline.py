from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
import numpy as np

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

text_clf = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	# ('clf', MultinomialNB())
	('clf', SGDClassifier())
	])

text_clf.fit(twenty_train.data, twenty_train.target)

# Testing

twenty_test = fetch_20newsgroups(
	subset='test', 
	categories = categories,
	shuffle = True)

predictions = text_clf.predict(twenty_test.data)
print(np.mean(predictions == twenty_test.target))

# Parameter tuning using grid search