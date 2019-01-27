from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

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

# Bag of words

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(twenty_train.data)

tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_count)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Test on two very simple documents.
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)

print(clf.predict(X_new_tfidf))