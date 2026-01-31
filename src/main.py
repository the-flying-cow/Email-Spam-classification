import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from load_data import get_data
from preprocessing import preprocess
from classifiers import train_svm, train_log
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
from build_logger import get_logger



def main():
    logger= get_logger(__name__)

    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PATH= kagglehub.dataset_download("advaithsrao/enron-fraud-email-dataset")
        dataset= os.listdir(PATH)
        FULL_PATH= os.path.join(PATH, dataset[0])

    except Exception as e:
        logger.exception('Failed to download dataset')
        raise




    try:
    # load dataset
        logger.info('Loaded dataset')
        data= get_data(FULL_PATH)
        

    # preprocessing steps
        logger.info('Preprocessing data')    
        prep_data= preprocess(data)
        
    except Exception as e:
        logger.exception('Failed to get dataset')
        raise


    
    
    try:
    # get target and features 
        target= prep_data['Label']
        features= prep_data['Text']
        feat_train, feat_test, targ_train, targ_test= train_test_split(features, target, test_size= 0.2, random_state= 7, stratify= target)
    
    except Exception as e:
        logger.exception('Failed to get target and features')
        raise



    
    try:
    # model training
        start_time= time.time()
        svm_model= train_svm(feat_train, targ_train)
        train_time= time.time() - start_time
        logger.info(f'SVM  Model Training Time: {train_time:.2f} seconds')
    

        start_time= time.time()
        log_model= train_log(feat_train, targ_train)
        train_time= time.time() - start_time
        logger.info(f' Logistic Model Training Time: {train_time:.2f} seconds')
    
    except Exception as e:
        logger.exception('Model Training failed')
        raise

    
    try:
    # save trained models
        model_dir= os.path.join(BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok= True)
        
        svm_path= os.path.join(model_dir, "svm_model.pkl")
        joblib.dump(svm_model, svm_path)
        
        log_path= os.path.join(model_dir, "log_model.pkl")
        joblib.dump(log_model, log_path)
    
    except Exception as e:
        logger.exception(e)
        raise
    

    
    # calculating model performance
    logger.info('---SVM Predictions---')
    svm_preds= svm_model.predict(feat_test)
    logger.info(f'Precision Score: {precision_score(targ_test, svm_preds)}\nRecall Score: {recall_score(targ_test, svm_preds)}\n\
                    F1 Score: {f1_score(targ_test, svm_preds)}\nConfusion Matrix: {confusion_matrix(targ_test, svm_preds)}')


    logger.info('---Logistic Regression Predictions---')
    log_preds= log_model.predict(feat_test)
    logger.info(f'Precision Score: {precision_score(targ_test, log_preds)}\nRecall Score: {recall_score(targ_test, log_preds)}\n\
                    F1 Score: {f1_score(targ_test, log_preds)}\nConfusion Matrix: {confusion_matrix(targ_test, log_preds)}')


if __name__=='__main__':
    main()