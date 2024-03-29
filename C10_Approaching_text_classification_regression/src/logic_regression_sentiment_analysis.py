import pandas as pd 

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

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
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for f, (t_,v_) in enumerate(kf.split(X = df, y=y)):
        df.loc[v_, 'kfold'] = f
    # we go over the folds created
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop = True)
        test_df = df[df.kfold ==fold_].reset_index(drop = True)
        # initialize CountVectorizer with NLTK's word_tokenize
        # function as tokenizer
        count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
        # fit count_vec on training data reviews
        count_vec.fit(train_df.review)

        # transform training and validation data reviews
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review) #new words?

        # initialize logistic regression model
        model = linear_model.LogisticRegression()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)

        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)

        # calculate accuracy
        accuracy = metrics.accuracy_score( test_df.sentiment, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")

        # Fold: 0
        # Accuracy = 0.8948
        # Fold: 1
        # Accuracy = 0.8941
        # Fold: 2
        # Accuracy = 0.8899
        # Fold: 3
        # Accuracy = 0.8942




        



