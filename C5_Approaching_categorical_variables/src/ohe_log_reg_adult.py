#The simplest model to build is by one hot encoding and logistic regression

import pandas as pd 
import numpy as np

import config

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv(config.TRAINING_FILE_ADULT)

    #list of numerical columns
    num_cols = [ "fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]

    #drop numerical columns
    df = df.drop(num_cols, axis = 1)

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
    # get training data using folds    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    # transform training data
    x_train = ohe.transform(df_train[features]) 

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize Logistic Regression model
    #Logistic Regression works with one hot encoder
    model = linear_model.LogisticRegression()

    #fit model on training data(ohe)
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
    #output:
    # Fold = 0, AUC = 0.883986048459122
    # Fold = 1, AUC = 0.8703360933268939
    # Fold = 2, AUC = 0.8766765580625453
    # Fold = 3, AUC = 0.8884736729946834
    # Fold = 4, AUC = 0.8747898658856745