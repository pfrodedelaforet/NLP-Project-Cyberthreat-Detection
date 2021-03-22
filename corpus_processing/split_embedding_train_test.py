from tqdm import tqdm
import numpy as np
"""This is the splitting between train and test set for the Masked-Language Modelling task : we select randomly the lines in the train/test set."""
f = open('../storage/cybersec_final.txt', 'r', encoding='utf-8')
g = open('../storage/cybersec_train.txt', 'a', encoding='utf-8')
h = open('../storage/cybersec_test.txt', 'a', encoding='utf-8')
for line in tqdm(f): 
    if line not in ['', '\n']:
        p = np.random.rand()
        if p < 1/3 :
            print(line, file = h)
        else : 
            print(line, file = g)

f.close()
g.close()
h.close()
