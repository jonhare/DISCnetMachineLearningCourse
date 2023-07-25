categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

from sklearn.datasets import load_files
twenty_train = load_files('data/twenty_newsgroups/train', categories=categories, shuffle=True, random_state=42, encoding='latin1')

twenty_train.target_names

len(twenty_train.data)

len(twenty_train.filenames)

print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])

twenty_train.target[:10]

for t in twenty_train.target[:10]:
	print(twenty_train.target_names[t])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

count_vect.vocabulary_.get(u'algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(stop_words='english',max_df=0.5,min_df=2)
X_train_tfidf = tfidf_vect.fit_transform(twenty_train.data)
X_train_tfidf.shape

#<EX1>
#...
#</EX1>

from sklearn.cluster import KMeans
km = KMeans(4)
km.fit(X_train_tfidf)

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vect.get_feature_names_out()
for i in range(4):
	print("Cluster %d:" % i, end="")
	for ind in order_centroids[i, :10]:
		print(' %s' % terms[ind], end="")
	print()

from sklearn import metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(twenty_train.target, km.labels_))

#<EX2>
for i in range(0,len(twenty_train.filenames)):          
    print(twenty_train.filenames[i] + " " + str(km.labels_[i]))
#</EX2>

#<EX3>
#...
#</EX3>

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3).fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_tfidf = tfidf_vect.transform(docs_new)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
	print('%r => %s' % (doc, twenty_train.target_names[category]))


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', KNeighborsClassifier(n_neighbors=3))])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)



import numpy as np
twenty_test = load_files('data/twenty_newsgroups/test', categories=categories, shuffle=True, random_state=42, encoding='latin1')
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, tol=None, random_state=42))])
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)


from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

metrics.confusion_matrix(twenty_test.target, predicted)

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', KNeighborsClassifier(n_neighbors=3))])
from sklearn.model_selection import GridSearchCV
parameters = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__n_neighbors': (1, 3, 5, 7)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

twenty_train.target_names[gs_clf.predict(['God is love'])[0]]

gs_clf.best_score_                                  

for param_name in sorted(parameters.keys()):
	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))




