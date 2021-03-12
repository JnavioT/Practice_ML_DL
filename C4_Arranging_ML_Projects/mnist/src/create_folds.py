import pandas as pd
from sklearn import model_selection
import os

this_file_path = os.path.abspath(__file__)
BASE_DIR =  os.path.dirname(this_file_path)
BASE_DIR2 =  os.path.dirname(BASE_DIR)

DATA_ADDRESS = os.path.join(BASE_DIR2, "input","mnist_train.csv")
DATA_CONVERT_ADDRESS = os.path.join(BASE_DIR2,"input","mnist_train_folds.csv")

##ok to ubuntu but other os ? -> use absolute path
#df = pd.read_csv("../input/mnist_train.csv")
df = pd.read_csv(DATA_ADDRESS)

df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
kf = model_selection.KFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df)):
    df.loc[val_, 'kfold'] = fold

#df.to_csv("../input/mnist_train_folds.csv", index=False)
df.to_csv(DATA_CONVERT_ADDRESS, index=False)