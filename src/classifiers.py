import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from build_logger import get_logger

logger = get_logger(__name__)

text_feat = 'Text'

# set model and vectorizer configurations
tfidf_config = {
    'ngram_range': (1, 2),
    'max_features': 2000,
    'min_df': 2,
    'max_df': 0.85,
    'norm': 'l2',
    'lowercase': True,
}

lgbm_config = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 6,
    'random_state': 7,
    'n_estimators': 500,
    'n_jobs': -1,
    'verbose': -1
}

catboost_config = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'random_seed': 7,
    'verbose': 0,
    'thread_count': -1,
}

engineered_feat = [
    'subject_len', 'body_len', 'text_len',
    'subject_word_count', 'body_word_count', 'text_word_count',
    'exclamation_count', 'question_count', 'url_count', 'digit_count',
    'uppercase_ratio', 'contains_reply', 'contains_forward', 'has_html', 'mean_word_len'
]

def train_stacking_classifier(x_train: pd.DataFrame, y_train: pd.Series):
    try:
        preprocessor = ColumnTransformer([
            ('tfidf', TfidfVectorizer(**tfidf_config), text_feat),
            ('scaler', StandardScaler(), engineered_feat),
        ])

        lgbm_pipe = Pipeline([
            ('lgbm_preprocessor', preprocessor),
            ('lgbm_estimator', LGBMClassifier(**lgbm_config))
        ])

        catboost_pipe = Pipeline([
            ('catboost_preprocessor', preprocessor),
            ('catboost_estimator', CatBoostClassifier(**catboost_config))
        ])

        base_lrnrs = [
            ('lgbm', lgbm_pipe),
            ('catboost', catboost_pipe)
        ]

        meta_estim = LogisticRegression(
            C=1.0,
            penalty='l2',
            random_state=7,
        )

        stack_model = StackingClassifier(
            estimators=base_lrnrs,
            final_estimator=meta_estim,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )

        return stack_model.fit(x_train, y_train)

    except Exception as e:
        logger.exception('Stacking classifier training failed.')
        raise
