#The simplest model to build is by one hot encoding and logistic regression

import pandas as pd 
import xgboost as xgb 
import numpy as np 
import config

from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv(config.TRAINING_FILE_ADULT)

    #list of numerical columns
    num_cols = [ "fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    #drop numerical columns
    #df = df.drop(num_cols, axis = 1)

    #map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    #all columns are features except income and kfold
    features = [f for f in df.columns if f not in ("income", "kfold")]

    # Replace ? with NaN
    df.replace('?',np.NaN, inplace= True)
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn’t matter because all are categories
    for col in features:
        df[col].fillna("NONE", inplace = True)
        df.loc[:, col] = df[col].astype(str)

    # now its time to label encode the features
    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()

        # fit label encoder on all data
        lbl.fit(df[col])

        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # transform training data
    x_train = df_train[features].values

    # transform validation data
    x_valid = df_valid[features].values

    # initialize random forest model
    model = xgb.XGBClassifier(n_jobs=-1)

    #fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    # run function for fold = 0
    # we can just replace this number and
    # run this for any fold
    for fold_ in range(5):
        run(fold_)
    #run code: python lbe_rf.py

    # Fold = 0, AUC = 0.8850187749705561
    # Fold = 1, AUC = 0.870412329828776
    # Fold = 2, AUC = 0.8747792882322831
    # Fold = 3, AUC = 0.8859718644739449
    # Fold = 4, AUC = 0.8735062934457763

    # Fold = 0, AUC = 0.9192530593765533
    # Fold = 1, AUC = 0.9134643466002906
    # Fold = 2, AUC = 0.9120823503855425
    # Fold = 3, AUC = 0.9241652554529093
    # Fold = 4, AUC = 0.9119733489573015
 
    # Fold = 0, AUC = 0.9297572583489576
    # Fold = 1, AUC = 0.9240740554413513
    # Fold = 2, AUC = 0.9213457948409947
    # Fold = 3, AUC = 0.9339630648982895
    # Fold = 4, AUC = 0.9253197676218545