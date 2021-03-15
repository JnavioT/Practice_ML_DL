
import pandas as pd
import config

from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
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

    # initialize Truncated SVD
    # we are reducing the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)

    # fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transform sparse training data
    x_train = svd.transform(x_train)

    # transform sparse validation data
    x_valid = svd.transform(x_valid)

    # initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)

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
    #run code: python ohe_svd_rf.py
    # Fold = 0, AUC = 0.7074194154928289
    # Fold = 1, AUC = 0.7062948829822087
    # Fold = 2, AUC = 0.7080074903573167
    # Fold = 3, AUC = 0.7039157865465633
    # Fold = 4, AUC = 0.7065240467759517


