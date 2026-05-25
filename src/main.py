import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from load_data import get_data
from preprocessing import preprocess
from classifiers import train_stacking_classifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import time
from build_logger import get_logger


def main():
    logger = get_logger(__name__)

    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PATH = kagglehub.dataset_download("advaithsrao/enron-fraud-email-dataset")
        dataset = os.listdir(PATH)
        FULL_PATH = os.path.join(PATH, dataset[0])

    except Exception as e:
        logger.exception('Failed to download dataset')
        raise

    try:
        logger.info('Loaded dataset')
        data = get_data(FULL_PATH)

        logger.info('Preprocessing data')
        prep_data = preprocess(data)
    except Exception as e:
        logger.exception('Failed to get dataset')
        raise

    try:
        target = prep_data['Label']
        feature_data = prep_data.drop(columns=['Label'])
        feat_train, feat_test, targ_train, targ_test = train_test_split(
            feature_data,
            target,
            test_size=0.2,
            random_state=7,
            stratify=target,
        )
    except Exception as e:
        logger.exception('Failed to get target and features')
        raise

    try:

        start_time = time.time()
        stacking_model = train_stacking_classifier(feat_train, targ_train)
        train_time = time.time() - start_time
        logger.info(f'Stacking Model Training Time: {train_time:.2f} seconds')
    except Exception as e:
        logger.exception('Model Training failed')
        raise

    try:
        # save models to 'models' directory
        model_dir = os.path.join(BASE_DIR, 'models')
        os.makedirs(model_dir, exist_ok=True)

        stacking_path = os.path.join(model_dir, 'stacking_model.pkl')
        joblib.dump(stacking_model, stacking_path)
    except Exception as e:
        logger.exception(e)
        raise

    # performance metrics for each model

    logger.info('--->Stacking Classifier Predictions<---')
    stacking_preds = stacking_model.predict(feat_test)
    logger.info(
        f'Precision Score: {precision_score(targ_test, stacking_preds):.2f}\n'
        f'Recall Score: {recall_score(targ_test, stacking_preds):.2f}\n'
        f'F1 Score: {f1_score(targ_test, stacking_preds):.2f}\n'
        f'Confusion Matrix: {confusion_matrix(targ_test, stacking_preds)}\n'
        f'Classification Report:\n{classification_report(targ_test, stacking_preds)}'
    )


if __name__ == '__main__':
    main()
