from pathlib import Path
import pandas as pd
import gensim
import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv
from operator import itemgetter
from tqdm import tqdm
from heapq import nlargest
"""This script aims at performing a clustering on the relevant articles using FastText and term-document matrices to represent documents."""
pca = decomposition.PCA()
store = pd.HDFStore("../storage/table_countvect_clustering/storage_embeddings.h5")
#df_train = store["df_tfidf_train"]
#df_test = store["df_tfidf_test"]
df_train = store["df_counts_train"]
df_test = store["df_counts_test"]
df = pd.concat([df_train, df_test])
filenames = []
with open('../storage/table_countvect_clustering/filenames.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         filenames.append(''.join(row))
mask = np.where(df['labels']==1)[0]
filenames = itemgetter(*mask)(filenames)
df = df[df.labels==1]
list_col_init = [x for x in df.columns if (np.count_nonzero(df[x]) > 1 and x != "labels")]
X = df[list_col_init]
wv = gensim.models.KeyedVectors.load(f'../storage/FastText_250/FastText_250.wordvectors', mmap = 'r')

word_embeddings = np.array([wv[x] for x in list_col_init])
termdoc = normalize(X.to_numpy(),'l1') 
c = np.einsum('ij,ki->kj', word_embeddings, termdoc)#multiply embedding of word i with the weight then sum 

"""
script of PCA to chose the number of components to keep in order to reduce computational power needed : 

pca.n_components = 200
d = pca.fit_transform(c)
percentage_var_exp = pca.explained_variance_/np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_exp)
print(cum_var_explained)
plt.plot(cum_var_explained)
plt.hlines(0.8, 0, 200, colors = 'r', label = '80 %')
plt.xlabel("n_components")
plt.ylabel("variance ratio explained")
plt.title('variance ratio PCA')
plt.savefig('../results/variance ratio explained')
"""

#Chosen : n_components = 21
pca.n_components = 21
reduced_array = pca.fit_transform(c)
"""
script to compare the values of k in terms of silhouette scores : 

liste_silh = []
for k in tqdm(range(2, 200)):
    liste = [[None] for _ in range(k)]
    labels = KMeans(n_clusters=k, random_state=0).fit_predict(reduced_array)
    silh = sklearn.metrics.silhouette_score(reduced_array, labels)
    liste_silh.append(silh)
    print(f"k={k}, mean silhouette score : {silh}")
    liste = [np.sum(termdoc[labels == i], axis = 0) for i in range(k)]
idxs = nlargest(15, enumerate(liste_silh), key=lambda x: x[1])
print(idxs)"""
#best_idx_count: [(2, 0.21176708170455133), (3, 0.19494200471216477), (5, 0.18073654694151395), (4, 0.1728376753617945), (193, 0.16783257980969254), (189, 0.16601446404560874), (188, 0.16553641964789267), (10, 0.1627785123298472), (186, 0.16177934463508653), (185, 0.16176553634470905), (9, 0.16153492702306127), (199, 0.16072127068419215), (197, 0.16008316540511167), (170, 0.15996906142233094), (196, 0.159809603943672)]
#best_idx_tfidf : [(2, 0.2159251344251667), (3, 0.20207988908457453), (199, 0.17808941944406895), (198, 0.17732681871481423), (197, 0.17588953396451515), (196, 0.17462590440533354), (193, 0.17332681281017814), (194, 0.17286145111896317), (195, 0.17223595611333795), (190, 0.1712795101446919), (191, 0.1710026885707554), (192, 0.1708564251136771), (189, 0.17084116514641667), (188, 0.17073907362688168), (187, 0.17072361881213943)]
"""Script to display the 10 most commonly used words in each cluster"""
k = 2
liste = [[None] for _ in range(k)]
labels = KMeans(n_clusters=k, random_state=0).fit_predict(reduced_array)
silh = sklearn.metrics.silhouette_score(reduced_array, labels)
print(f"k={k}, mean silhouette score : {silh}")
liste = [np.sum(termdoc[labels == i], axis = 0) for i in range(k)]
for i in range(k): 
    #je prends les plus fréquents, avec n_largest, puis je récupère à partir de ça le nom des mots, et enfin je print les 10 mots les plus fréquents par classe
    n_largest = nlargest(10, range(len(liste[i])), key=lambda x: liste[i][x])
    print(n_largest)
    words = list(itemgetter(*n_largest)(X.columns))
    print(f'cluster : {k}, most frequent : {words}')
