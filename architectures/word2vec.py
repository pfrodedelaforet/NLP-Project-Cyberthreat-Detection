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
embedding_corpus = open('../storage/treated/cybersec_final.txt', 'r', encoding='utf-8')
cores = multiprocessing.cpu_count()
print(cores)
text = embedding_corpus.read()
text = text.replace('\n\n','. ')
text = text.replace('..', '.')
avant0 = time()
data = split_into_lists_of_words(text)
delta0 = time()-avant0
print(delta0)
model = gensim.models.Word2Vec(min_count = 2, size = 200, window = 5, workers = cores-1) 
model.build_vocab(data)
model.train(data, total_examples = model.corpus_count, epochs = 30, report_delay=1)
model.wv.save("CBOW.wordvectors")
wv = gensim.models.KeyedVectors.load('CBOW.wordvectors', mmap = 'r')

list_words = np.random.choice(list(model.vocab.keys()), 800)
word_vectors_arr = np.array([wv[word] for word in list_words])
twodim = PCA().fit_transform(word_vectors_arr)[:, :2]
plt.figure(figsize = (6,6))
plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
for word, (x,y) in zip(list_words, twodim):
    plt.text(x+0.05, y+0.05, word)
plt.show()
