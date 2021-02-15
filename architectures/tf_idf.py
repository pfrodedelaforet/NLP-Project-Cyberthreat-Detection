from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
#PROBLEME : INCESTE, CAR ON UTILISE NOTAMMENT LES DONNEES DE TEST POUR SAVOIR SI C EST RARE OU NON
nltk.download('stopwords')
PATH = Path("../storage/dataset/treated_articles")
def tf_idf(path):
    filenames = []
    labels = []
    iterd = PATH.iterdir()
    for cat in iterd : 
        filenames += [str(x) for x in cat.iterdir()]
        labels += [(str(cat)[-17 :] == '\RELEVANT_TREATED') for x in cat.iterdir()]
    x_train, x_test, y_train, y_test = train_test_split(filenames, labels,  test_size=0.33, random_state=42, stratify = labels)
    vectorizer = TfidfVectorizer(input = 'filename', stop_words = stopwords.words('english'))
    vectors = vectorizer.fit_transform(x_train)
    vectors_test = vectorizer.transform(x_test)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    dense_test = vectors_test.todense()
    denselist = dense.tolist()
    denselist_test = dense_test.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df_test = pd.DataFrame(denselist_test, columns=feature_names)
    df["labels"] = y_train
    df_test["labels"] = y_test
    store = pd.HDFStore("../storage/storage_embeddings.h5")
    store["df_tfidf_train"] = df
    store["df_tfidf_test"] = df_test


if __name__ == "__main__":
    tf_idf(PATH)
    store = pd.HDFStore("../storage/storage_embeddings.h5")
    df = store["df_tfidf_train"]
    list_col = [x for x in df.columns if np.count_nonzero(df[x]) > 1]
    print(list_col)