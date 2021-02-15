import csv
from tqdm import tqdm
def chunks(liste, n):
    new_list = [None]*(len(liste)//n+1)
    k = 0
    for i in tqdm(range(0, len(liste), n)):
        new_list[k] = ' '.join(liste[i:i+n])
        k += 1
    print(k-len(new_list)+1)
    new_list[-1] = ' '.join(liste[len(liste)//n*n:])
    return new_list
f = chunks(open('../storage/cybersec_train.txt', 'r', encoding='utf-8').read().replace('\n\n','. ').replace('..', '.').replace('\n', '').split(' '), 1024)
g = chunks(open('../storage/cybersec_test.txt', 'r', encoding='utf-8').read().replace('\n\n','. ').replace('..', '.').replace('\n', '').split(' '), 1024)

with open('../storage/block_emb/cybersec_train.txt', 'a') as train_file:
    for line in f : 
        print(line, file = train_file)

with open('../storage/block_emb/cybersec_test.txt', 'a') as test_file:
    for line in g : 
        print(line, file = test_file)


f.close()
g.close()