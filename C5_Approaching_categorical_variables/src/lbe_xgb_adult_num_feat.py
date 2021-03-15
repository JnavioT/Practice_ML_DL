import itertools
import pandas as pd 
import xgboost as xgb 
import numpy as np 
import config

from sklearn import metrics
from sklearn import preprocessing

def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of values
    # in this list
    # for example:
    # list(itertools.combinations([1,2,3], 2)) will return
    # [(1, 2), (1, 3), (2, 3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE_ADULT)

    #list of numerical columns
    #drop_cols = ["fnlwgt"]
    #df = df.drop(drop_cols, axis = 1)

    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    #drop numerical columns
    #df = df.drop(num_cols, axis = 1)

    #map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    # list of categorical columns for feature engineering
    cat_cols = [ c for c in df.columns if c not in num_cols and c not in ("kfold", "income") ]
    # add new features
    df = feature_engineering(df, cat_cols)

    #all columns are features except income and kfold
    features = [f for f in df.columns if f not in ("income", "kfold")]

    # Replace ? with NaN
    df.replace('?',np.NaN, inplace= True)
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn’t matter because all are categories
    for col in features:
        if col not in num_cols:
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

    # get training data using folds    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

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
    # run function for fold = 0
    # we can just replace this number and
    # run this for any fold
    for fold_ in range(5):
        run(fold_)
    # output:

    # Fold = 0, AUC = 0.9197995239480138
    # Fold = 1, AUC = 0.9132111923956475
    # Fold = 2, AUC = 0.9116305426903771
    # Fold = 3, AUC = 0.9238127100051186
    # Fold = 4, AUC = 0.9121721959414834

    # Fold = 0, AUC = 0.9293645876519375
    # Fold = 1, AUC = 0.9260671046331154
    # Fold = 2, AUC = 0.9248297126799749
    # Fold = 3, AUC = 0.9343539865885674
    # Fold = 4, AUC = 0.9248902761954295
