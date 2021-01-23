from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import numpy as np

import joblib

# Load
dataset = fetch_20newsgroups(data_home="../data", subset="all", remove=("headers", "footers"))

# Extract
data = dataset['data']
targets = dataset['target']
classes = dataset['target_names']
index2name = dict(zip(range(len(classes)), classes))
class_targets = [index2name[m] for m in targets]

vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=0.01, max_features=15000, lowercase=True)
classifier = LogisticRegression(random_state=42, max_iter=150, verbose=1, n_jobs=-1)
pipe = make_pipeline(vect, classifier)

# Transform/Train
cv_results = cross_validate(pipe, data, class_targets, verbose=1, return_estimator=True, return_train_score=True,
                            scoring=["accuracy"], n_jobs=-1, cv=5)

best_test_acc = np.argmax(cv_results['test_accuracy'])
best_estimator = cv_results['estimator'][best_test_acc]


joblib.dump(best_estimator, "trained_20newsgroups.joblib")
print("Done")
