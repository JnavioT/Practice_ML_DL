import pandas as pd 
from sklearn import model_selection
import os

this_file_path = os.path.abspath(__file__)
BASE_DIR =  os.path.dirname(this_file_path)
BASE_DIR2 =  os.path.dirname(BASE_DIR)

DATA_ADDRESS = os.path.join(BASE_DIR2, "input","train.csv")
DATA_CONVERT_ADDRESS = os.path.join(BASE_DIR2,"input","cat_train_folds.csv")

if __name__ == "__main__":
    df = pd.read_csv(DATA_ADDRESS)
    #we create a new column called kfold and fill it with -1
    df['kfold'] = -1
    #randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop = True)
    # fetch labels
    y = df.target.values
    #init the kfold class from model_selection
    #we saw in exploration.ipynb that is a binary classification problem with skewed targets.
    #So we will use StratifiedKFold
    kf = model_selection.StratifiedKFold(n_splits=5)
    #If we use only kfold the data per fold will have different distribution between classes
    #kf = model_selection.KFold(n_splits=5)
    #fill the new kfold column
    for f,(t_, v_) in enumerate(kf.split(X = df, y = y)):
        df.loc[v_, 'kfold'] = f
    #save the new csv with kfold column
    df.to_csv(DATA_CONVERT_ADDRESS, index = False)