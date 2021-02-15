from transformers import BertTokenizer, RobertaTokenizer
import sys
from megasplittor2000 import split_into_lists_of_words
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from operator import itemgetter

PATH = Path("../storage/treated_articles")
iterd = PATH.iterdir()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_length = 512)
text_art = ''
for label in tqdm(iterd):#on peut bien modifier iterdir c'est pas grave
    for article in tqdm(label.iterdir()):
        text_art += open(str(article), 'r', encoding='utf-8').read().replace('\n\n', '. ').replace('..', '.')
def chunks(liste, n):
    for i in range(0, len(liste), n):
        yield liste[i:i+n]
    yield liste[len(liste)//n*n:]
text = open('../storage/cybersec_final.txt', 'r', encoding='utf-8').read()
#object_methods = [method_name for method_name in dir(RobertaTokenizer)
                #if callable(getattr(RobertaTokenizer, method_name))]
#print(object_methods)
text = text.replace('\n\n', '. ')
text = text.replace('..', '.')
text = text.replace('#', '')
text_art = text_art.replace('\n\n', '. ')
text_art = text_art.replace('..', '.')
text_art = text_art.replace('#', '')
data_art = split_into_lists_of_words(text_art)
data_emb = split_into_lists_of_words(text)
data_art = [x for sent in data_art for x in sent]
data_emb = [x for sent in data_emb for x in sent]

c = Counter(data_art)
d = Counter(data_emb)
data_art =  set(x for x in c.keys() if c[x] > 5)
data_emb = set(map(itemgetter(0), d.most_common(15000)))
print(len(tokenizer))
data = list(data_art | data_emb)
print(len(data), len(tokenizer))
for x in tqdm(chunks(data, 1000)):
    tokenizer.add_tokens(x)
tokenizer.save_pretrained('../storage/tokenizer_roberta')