import sys, pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LinearSVC, Perceptron
# from sklearn.pipeline import Pipeline

test_text = sys.argv[1]

with open('./trained_en', 'rb') as file:
	clf_en = pickle.load(file)
with open('./trained_review', 'rb') as f:
	gs_review = pickle.load(f)

print(clf_en)
print(gs_review)

predict_en = clf_en.predict([test_text])
print(predict_en)

if predict_en == 2:
	predict_review = gs_review.predict([test_text])
	print(predict_review)