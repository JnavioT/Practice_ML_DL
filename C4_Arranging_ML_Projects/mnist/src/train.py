import joblib
import pandas as pd 
import os
from sklearn import metrics
from sklearn import tree
import config


def run(fold):
    #read the data with folds:
    #df = pd.read_csv("../input/mnist_train_folds.csv")
    df = pd.read_csv(config.DATA_ADD)

    #training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label",axis = 1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop("label",axis = 1).values
    y_valid = df_valid.label.values

    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()

    # fit the model on training data
    clf.fit(x_train,y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid,preds)
    print(f"Fold = {fold}, Accuracy = {accuracy}")

    #save the model
    #joblib.dump(clf,f"../models/dt_{fold}.bin")
    joblib.dump(clf,os.path.join(config.MODELS_ADD ,f"dt_{fold}.bin"))

if __name__ == "__main__":
    for i in range(5):
        run(fold = i)


