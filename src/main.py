import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from load_data import get_data
from preprocessing import preprocess
from classifiers import train_svm, train_log

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH= kagglehub.dataset_download("advaithsrao/enron-fraud-email-dataset")
dataset= os.listdir(PATH)
FULL_PATH= os.path.join(PATH, dataset[0])
def main():
    
    data= get_data(FULL_PATH)

    print("Preprocessing data...")
    prep_data= preprocess(data)

    target= prep_data['Label']
    features= prep_data['Text']
    feat_train, feat_test, targ_train, targ_test= train_test_split(features, target, test_size= 0.2, random_state= 7, stratify= target)

    print("Training SVM Model...")
    svm_model= train_svm(feat_train, targ_train)
    print("Training Log Model...")
    log_model= train_log(feat_train, targ_train)

    model_dir= os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok= True)
    svm_path= os.path.join(model_dir, "svm_model.pkl")
    joblib.dump(svm_model, svm_path)
    print("SVM Model saved")

    log_path= os.path.join(model_dir, "log_model.pkl")
    joblib.dump(log_model, log_path)
    print("Log Model saved")

    return log_model

if __name__=='__main__':
    main()