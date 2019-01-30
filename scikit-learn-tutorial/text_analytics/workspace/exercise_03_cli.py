import sys, pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LinearSVC, Perceptron
# from sklearn.pipeline import Pipeline

test_text = sys.argv[1]

with open('./trained_en', 'rb') as file:
	clf_en = pickle.load(file)
with open('./trained_review', 'rb') as f:
	gs_review = pickle.load(f)

target_names = clf_en[1]

predict_en = clf_en[0].predict([test_text])
lang = target_names[predict_en[0]]

if lang == 'en':
	predict_review = gs_review.predict([test_text])
	if predict_review[0] == 0:
		print('Negative review written in English.')
	elif predict_review[0] == 1:
		print('Positive review written in English.')
	else:
		print('Failed to classifying review written in English.')
else:
	print('English not detected.')