
import copy
import pandas as pd 

import numpy as np 
import config

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb 

def mean_target_encoding(data):
    # make a copy of dataframe
    df = copy.deepcopy(data)
    # list of numerical columns
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    # all columns are features except income and kfold columns
    features = [f for f in df.columns if f not in ("kfold", "income") and f not in num_cols ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesnt matter because all are categories

    df.replace('?',np.NaN, inplace= True)

    for col in features:
        # do not encode the numerical columns
        if col not in num_cols:
            #df.loc[:, col] = df[col].astype(str).fillna("NONE")
            df[col].fillna("NONE", inplace = True)
            df.loc[:, col] = df[col].astype(str)
    # now its time to label encode the features
    for col in features:
        if col not in num_cols:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            # fit label encoder on all data
            lbl.fit(df[col])
            # transform all the data
            df.loc[:, col] = lbl.transform(df[col])
    # a list to store 5 validation dataframes
    encoded_dfs = []

    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # for all feature columns, i.e. categorical columns
        for column in features:
            # create dict of category:mean target
            mapping_dict = dict( df_train.groupby(column)["income"].mean())
            # column_enc is the new column we have with mean encoding
            df_valid.loc[:, column + "_enc"] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df

def run(df, fold):
    # note that folds are same as before
    # get training data using folds   
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # all columns are features except income and kfold columns
    features = [f for f in df.columns if f not in ("kfold", "income")]   
    # transform training data
    x_train = df_train[features].values

    # transform validation data
    x_valid = df_valid[features].values

    # initialize random forest model
    model = xgb.XGBClassifier(n_jobs=-1,max_depth= 7)

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
    df = pd.read_csv(config.TRAINING_FILE_ADULT)
    # create mean target encoded categories and
    # merge data
    df = mean_target_encoding(df)

    for fold_ in range(5):
        run(df, fold_)
    # output:

    # Fold = 0, AUC = 0.9296497444417628
    # Fold = 1, AUC = 0.925812337981309
    # Fold = 2, AUC = 0.9254923639662176
    # Fold = 3, AUC = 0.9355846062636218
    # Fold = 4, AUC = 0.9266719013151377
