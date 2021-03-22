from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(1, 'C:/Users/pfrod/OneDrive/Documents/2A mines de paris/Data x cybersec/cours_data')
from eval import Performance, show_top10
import numpy as np
"""Implementation of the Multinomial Naive Bayes, just change "counts" to "tfidf" if you want to change the weighting."""

if __name__ == '__main__':
    PATH = Path("../storage/dataset/treated_articles")
    store = pd.HDFStore("../storage/storage_embeddings.h5")
    df_train = store["df_counts_train"]
    list_col = [x for x in df_train.columns if np.count_nonzero(df_train[x]) > 2]
    print(len(list_col), len(df_train.columns))
    y_train = df_train["labels"].to_numpy()
    X_train = df_train[list_col].drop(columns = ['labels']).to_numpy()
    df_test = store["df_counts_test"]
    y_test = df_test["labels"].to_numpy()
    X_test = df_test[list_col].drop(columns = ['labels']).to_numpy()
    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)
    y_pred = MNB.predict(X_test)
    print(np.sum((1-y_pred)*(1-y_test))/np.sum(1-y_pred))
    perf = Performance()
    print(f'accuracy : {perf.accuracy(y_pred, y_test)},\n precision : {perf.precision(y_pred, y_test)},\n recall : {perf.recall(y_pred, y_test)},\n F_score : {perf.F_score(y_pred, y_test)}')
    print(show_top10(MNB, list_col))

