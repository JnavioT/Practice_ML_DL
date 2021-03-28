import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    df =  pd.read_csv("../input/imdb.csv")
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    # we create a new column called kfold and fill it with -1
    df['kfold'] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch labels
    y = df.sentiment.values
    # initiate the kfold class from model_selection module
    #kf = model_selection.StratifiedKFold(n_splits=5)
    kf = model_selection.KFold(n_splits=5)
    # fill the new kfold column
    for f, (t_,v_) in enumerate(kf.split(X = df, y=y)):
        df.loc[v_, 'kfold'] = f
    # we go over the folds created
    for fold_ in range(5):
        # temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop = True)
        test_df = df[df.kfold == fold_].reset_index(drop = True)

        tfidf_vec = TfidfVectorizer(tokenizer = word_tokenize, token_pattern = None)

        tfidf_vec.fit(train_df.review)

        xtrain = tfidf_vec.transform(train_df.review)
        xtest = tfidf_vec.transform(test_df.review)

        model = linear_model.LogisticRegression()
        model.fit(xtrain, train_df.sentiment)

        preds = model.predict(xtest)

        accuracy = metrics.accuracy_score(test_df.sentiment, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")

        # Fold: 0
        # Accuracy = 0.8996

        # Fold: 1
        # Accuracy = 0.8952
        
        # Fold: 2
        # Accuracy = 0.8933

        # Fold: 3
        # Accuracy = 0.8948

        # Fold: 4
        # Accuracy = 0.9033

