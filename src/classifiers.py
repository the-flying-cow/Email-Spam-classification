import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

tfidf= TfidfVectorizer(ngram_range= (1,2), max_features=2000, lowercase= True, min_df= 2, max_df= 0.85, norm= 'l2')

def train_svm(x_train: pd.DataFrame, y_train: pd.DataFrame) -> TunedThresholdClassifierCV:
    model= Pipeline(steps=[("tfidf", tfidf), ("svm_classifier", LinearSVC(C= 0.90, max_iter= 3000, penalty='l1', dual= False, random_state=7))])
    tuned_svm= TunedThresholdClassifierCV(estimator= model, scoring= 'f1',cv= 3, random_state= 7).fit(x_train, y_train)
    return tuned_svm

def train_log(x_train: pd.DataFrame, y_train: pd.DataFrame) -> TunedThresholdClassifierCV:
    model= Pipeline(steps=[("tfidf", tfidf), ("log_classifier", LogisticRegression(C= 0.90, solver= 'saga', penalty= 'l1', dual= False, random_state= 7, n_jobs= -1, max_iter= 3000))])
    tuned_log= TunedThresholdClassifierCV(estimator= model, scoring= 'f1',cv= 3, random_state= 7).fit(x_train, y_train)
    return tuned_log