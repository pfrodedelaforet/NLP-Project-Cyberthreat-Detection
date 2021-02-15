import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.models import Word2Vec
from time import time
from megasplittor2000 import split_into_lists_of_words
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
list_words = ['information', 'vulnerabilities', 'vuln', 'attacks', 'data', 'malwapi', 'ransom', 'security', 'cve', 'malspam', 
    'code','politics', 'company', 'knight', 'kitchen', 'submarine', 'good', 'student', 'neighbor', 'pineapple']
#df = pd.DataFrame(columns = list_words)
list_plot = []
for size in [25, 50, 100, 150, 200, 250, 300]:
    wv = gensim.models.KeyedVectors.load(f'../storage/FastText_{size}/FastText_{size}.wordvectors', mmap = 'r')
    list_plot.append(np.mean(np.var(np.array([list([x[1] for x in wv.most_similar(word, restrict_vocab = 5000)]) for word in list_words]), axis = 1)))
    #print(df)
    #new_df = pd.DataFrame(np.vstack([np.array([size]*len(list_words)), np.array([list([x[0] for x in wv.most_similar(word, restrict_vocab = 5000)]) for word in list_words]).T]), columns = list_words)
    #df = df.append(new_df, ignore_index = True)
#df.to_csv('../resultats_proches_voisins.csv')
plt.plot([25, 50, 100, 150, 200, 250, 300], list_plot)
plt.xlabel("Embedding_size")
plt.ylabel("Mean similarity variance")
plt.savefig('../variance_similarities')
"""If I want visualization
word_vectors_arr = np.array([wv[word] for word in list_words])
twodim = PCA().fit_transform(word_vectors_arr)[:, :2]
plt.figure(figsize = (6,6))
plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
for word, (x,y) in zip(list_words, twodim):
    plt.text(x+0.05, y+0.05, word)"""