#The simplest model to build is by one hot encoding and logistic regression

import pandas as pd 
import xgboost as xgb 
import config

from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    #all columns are features except id, target and kfold
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn’t matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

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
    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7, n_estimators=200)

    #fit model on training data
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    # run function for fold = 0
    # we can just replace this number and
    # run this for any fold
    for fold_ in range(5):
        run(fold_)
    #run code: python lbe_rf.py

    # Fold = 0, AUC = 0.7643052237692294
    # Fold = 1, AUC = 0.7661504887752599
    # Fold = 2, AUC = 0.7647351254723161
    # Fold = 3, AUC = 0.7642869940106419
    # Fold = 4, AUC = 0.7633906382125766