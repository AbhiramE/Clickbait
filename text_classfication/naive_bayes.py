import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


def load_data(fileName):
    dict_data = []
    for line in open(fileName, 'r'):
        dict_data.append(json.loads(line))
    return dict_data


def get_post_data(dict_data):
    post_data = []
    for dict in dict_data:
        post_data.append(dict['postText'][0].encode('utf-8').rstrip())
    return np.array(post_data)


def get_labels(labels_data):
    score_data = []
    for dict in labels_data:
        score_data.append(0 if dict['truthMean'] <= 0.5 else 1)
    return np.array(score_data)


def get_count_vector(post_data):
    print post_data
    count_vect = CountVectorizer()
    X_train_count = count_vect.fit_transform(post_data)
    print "Count Vector Done", X_train_count.shape
    return X_train_count


def get_tf_idf(X_train_count):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
    print "TF IDF Done", X_train_tfidf.shape
    return X_train_tfidf


def pipeline(X_train, y_train):
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False)}
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    gs_clf = GridSearchCV(text_clf, parameters)
    gs_clf = gs_clf.fit(X_train[:5000], y_train[:5000])

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=gs_clf.best_params_['vect__ngram_range'])),
                         ('tfidf', TfidfTransformer(use_idf=gs_clf.best_params_['tfidf__use_idf'])),
                         ('clf', MultinomialNB())])
    text_clf = text_clf.fit(X_train, y_train)
    return text_clf


def predict(clf, X_test, y_test):
    predicted = clf.predict(X_test)
    matched = 0
    for i in range(0, len(y_test)):
        if predicted[i] < 0.5 and y_test[i] < 0.5:
            matched += 1
        elif predicted[i] >= 0.5 and y_test[i] >= 0.5:
            matched += 1

    return float(matched) / len(y_test)


dict_data = load_data('../clickbait17-validation-170630/instances.json')
labels_data = load_data('../clickbait17-validation-170630/truth.json')

X = get_post_data(dict_data)
y = get_labels(labels_data)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_counts = get_count_vector(X_train)
X_train_tfidf = get_tf_idf(X_train_counts)

clf = pipeline(X_train, y_train)
print "Accuracy at ", predict(clf, X_validate, y_validate)
