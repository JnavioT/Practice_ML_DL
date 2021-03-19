import os

import numpy as np 
import pandas as pd 

from PIL import Image 
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm 

def create_dataset(training_df, image_dir):
    """
    This function takes the training dataframe
    and outputs training array and labels
    :param training_df: dataframe with ImageId, Target columns
    :param image_dir: location of images (folder), string
    :return: X, y (training array with features and labels)
    """
    # create empty list to store image vectors
    images = []
    # create empty list to store targets
    targets = []
    # loop over the dataframe
    for index,row in tqdm(training_df.iterrows(), total=len(training_df), desc="processing images" ):
        # get image id
        image_id = row["ImageId"]
        # create image path
        image_path = os.path.join(image_dir, image_id)
        #open image using PIL
        image = Image.open(image_path + ".png")
        # resize image to 256x256. we use bilinear resampling
        image = image.resize((256,256), resample = Image.BILINEAR)
        # convert image to array
        image = np.array(image)
        #ravel
        image = image.ravel()
        # append images and targets lists
        images.append(image)
        targets.append(int(row["target"]))
    # convert list of list of images to numpy array
    images = np.array(images)
    print(images.shape)
    return images, targets

if __name__ == "__main__":
    csv_path = "../input/train.csv"
    image_path = "../input/train_png/"
    #image_path_test = "../input/test_png/"

    # read csv with imageid and target columns
    df = pd.read_csv(csv_path)
    # create new column
    df['kfold'] = -1
    
    df = df.sample(frac=1).reset_index(drop =  True)
    y = df.target.values

    # initiate kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)

    #fill the new kfold column
    for f,(t_,v_) in enumerate(kf.split(X = df, y=y)):
        df.loc[v_,'kfold'] = f
    
    # go over folds
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop = True)
        test_df = df[df.kfold == fold_].reset_index(drop = True)

        #create train dataset
        xtrain, ytrain = create_dataset(train_df,image_path)

        #create test dataset
        xtest, ytest = create_dataset(test_df, image_path)

        # fit random forest by default
        clf = ensemble.RandomForestClassifier(n_jobs = -1)
        clf.fit(xtrain, ytrain)

        #predict probability of class 1
        preds = clf.predict_proba(xtest)[:,1]

        print(f"FOLD: {fold_}")
        print(f"AUC = {metrics.roc_auc_score( ytest, preds)}")
        print("")


    # FOLD: 0
    # AUC = 0.7189137603043755
    # FOLD: 1
    # AUC = 0.7204628686917804
    # FOLD: 2
    # AUC = 0.7311101655852213
    # FOLD: 3
    # AUC = 0.7283217084302076
    # FOLD: 4
    # AUC = 0.7349604398721513