import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    #read data
    df = pd.read_csv("../input/imdb.csv")
    # map text to 0 or 1
    df.sentiment = df.sentiment.apply(lambda x:1 if x == "positive" else 0)
    # create new column
    df['kfold'] = -1
    #randomize data
    df = df.sample(frac=1).reset_index(drop = True)
    #fetch labels
    y = df.sentiment.values
    #initialize kfold
    kf = model_selection.StratifiedKFold(n_splits = 5)
    #fill the new kfold column
    for f, (t_,v_) in enumerate(kf.split(X =df, y=y)):
        df.loc[v_, 'kfold'] = f
    #save the new csv
    df.to_csv("../input/imdb_folds.csv", index = False)


