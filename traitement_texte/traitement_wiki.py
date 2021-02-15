#attempt to collect a more general embedding dataset from wiki articles using cleaner, and custom cleaning.



"""récupération et premier nettoyage de dump
from wiki_dump_reader import Cleaner, iterate
from tqdm import tqdm
cleaner = Cleaner()
f = open('wiki.txt', 'w')
for title, text in tqdm(iterate('poubelle/enwiki-20201201-pages-articles-multistream.xml')):
    try : 
        text = cleaner.clean_text(text)
        cleaned_text, _ = cleaner.build_links(text)
        
        print(cleaned_text, file = f)
    except : 
        pass
f.close()"""
""" premier essai de second nettoyage de dump
from tqdm import tqdm
f = open("wiki.txt", "r")
i = 0
g = open("new_wiki.txt", 'w')
for line in tqdm(f) :
    if line[0:2]!= "==" and line[0:8] != 'REDIRECT':
        print(line, file = g)
f.close()
g.close()
"""

import re 
f = open("../storage/embedding/brut/wiki.txt", "r")
g = open("../storage/embedding/treated/new_wiki_2.txt", 'w')
line = next(f, "fin de traitement")
while line != "fin de traitement" :
    while line[0:8] == 'REDIRECT' and line != 'fin de traitement'and line[0] != '|':#tant qu'on n'est pas dans le corps de l'article
        line = next(f, "fin de traitement")
        p = False
    if (not p) and re.search('cyber|malware|internet|comput', line.lower()) : #si on vient d'arriver dans un article qui parle de ce qu'on veut
        p = True
        first_line = line
    while line[0:8] != 'REDIRECT' and line != 'fin de traitement' :
        if line[0:2] != '==' and p and len(line) > 80 :
            print(line, file = g)
            line = next(f, "fin de traitement")
        elif re.search("==See also==|==External links==|==References==", line):
            while len(line) < 120 :
                line = next(f, "fin de traitement")
            break
        else : 
            line = next(f, "fin de traitement")
    p = False

f.close()
g.close()