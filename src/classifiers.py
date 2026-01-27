import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from build_logger import get_logger

logger= get_logger(__name__)

tfidf_config= {
    "ngram_range": (1, 2),
    "max_features": 2000,
    "min_df": 2,
    "max_df": 0.85,
    "norm": "l2",
    "lowercase": True
}



svm_config= {
    "C": 0.9,
    "penalty": "l1",
    "dual": False,
    "max_iter": 3000,
    "random_state": 7
}




log_config= {
    "C": 0.9,
    "solver": 'saga',
    "penalty": 'l1',
    "dual": False, 
    "random_state": 7,
    "n_jobs": -1, 
    "max_iter":3000
}


def train_svm(x_train: pd.DataFrame, y_train: pd.Series) -> TunedThresholdClassifierCV:
    
    try:
        
        logger.info('TfIdfVectorizer Hyperparameters')
        tfidf= TfidfVectorizer(**tfidf_config)
        logger.info(tfidf_config)
        logger.info('SVM Model Hyperparameters')
        model= Pipeline(steps=[("tfidf", tfidf),
                            ("svm_classifier", LinearSVC(**svm_config))])
        logger.info(svm_config)

        logger.info('ThresholdTuner Hyperparameters')
        threshold_tuning_config= {
            "estimator": model,
            "scoring": 'f1', # using f1 as scoring
            "cv": 3,
            "random_state": 7
            }
        tuned_svm= TunedThresholdClassifierCV(**threshold_tuning_config).fit(x_train, y_train)
        logger.info(f"Optimal decision threshold: {tuned_svm.best_threshold_}")

    
    except Exception as e:
        logger.exception('SVM training failed.')
        raise

    return tuned_svm




def train_log(x_train: pd.DataFrame, y_train: pd.Series) -> TunedThresholdClassifierCV:
    try:

        logger.info('TfIdfVectorizer Hyperparameters')
        tfidf= TfidfVectorizer(**tfidf_config)
        logger.info(tfidf_config)

        
        logger.info('Logistic Model Hyperparameters')
        model= Pipeline(steps=[("tfidf", tfidf),
                            ("log_classifier", LogisticRegression(**log_config))])
        logger.info(log_config)

        logger.info('ThresholdTuner Hyperparameters')
        threshold_tuning_config= {
            "estimator": model,
            "scoring": 'precision', # using precision as scoring
            "cv": 3,
            "random_state": 7
            }
        tuned_log= TunedThresholdClassifierCV(**threshold_tuning_config).fit(x_train, y_train)
        logger.info(f"Optimal decision threshold: {tuned_log.best_threshold_}")

    
    except Exception as e:
        logger.exception('Logistic training failed.')
        raise

    return tuned_log