import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.models import FastText
from time import time
from megasplittor2000 import split_into_lists_of_words
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import logging
from tqdm import tqdm
embedding_corpus = open('../storage/cybersec_final.txt', 'r', encoding='utf-8')
cores = multiprocessing.cpu_count()
print(cores)
text = embedding_corpus.read()
text = text.replace('\n\n','. ')
text = text.replace('..', '.')
debut = time()
data = split_into_lists_of_words(text)
print(time()-debut)
def fasttext_embedding(vector_size):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.FastText(min_count = 5, size = vector_size, window = 5, workers = cores, min_n = 3, max_n = 6) 
    model.build_vocab(data)
    model.train(data, total_examples = model.corpus_count, epochs = 10, report_delay=1)
    model.save(f"../storage/FastText_{size}/FastText_{size}.model")
    model.wv.save(f"../storage/FastText_{size}/FastText_{size}.wordvectors")

if __name__ =='__main__':
    for size in tqdm([25, 50, 100, 150, 200, 250, 300]):
        fasttext_embedding(size)