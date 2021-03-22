from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions
import os
from pathlib import Path
from tqdm import tqdm
import re 
"""Approximately same thing as clean_dataset_text, but applied on embedding data.This treatment follows the embedding_treatment.py, in which we adapted the treatment to each website.
Here it is a final global treatment applied to the whole corpus."""
cequongarde = [',', '.', '"', ':', ')', '(','!', '?' ';']
puncts = ['-','|', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', ',', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
article = open('../storage/treated/all_treated.txt', 'r', encoding = 'utf-8')
treated_article = open('../storage/treated/cybersec_final_heavy_articles.txt', 'a', encoding = 'utf-8')
for line in tqdm(article) : 
    if line != '':
        if len(line) < 100 or 'cooki' in line or 'Privacy Settings' in line or 'NOTE' in line: 
            continue
        new_line = unidecode.unidecode(line)#remove accents
        if new_line[1] in [")", "."] :
            new_line = new_line[2:]
        if new_line[0] in ["•", '-', '*']:
            new_line = new_line[1:]
        for el in cequongarde : 
            if el in new_line :  
                new_line = new_line.replace(el,' '+el+' ')#add spaees before and after characters we want to keep, no worries if there is too much because next function fixes it. 
        new_line = contractions.fix(new_line)#change n't to not and 're to are
        new_line = re.sub(r'http\S+', '', new_line)#remove urls
        for punct in puncts :
            if punct in new_line : 
                new_line = new_line.replace(punct, '')#remove useless punct
        if bool(re.search(r'\d', new_line)):
            new_line = re.sub(r'\d', '#', new_line)
            new_line = new_line.replace("#,#", '#')
            new_line = new_line.replace("#.#","#")
            new_line = new_line.replace("#,#", '#')
            new_line = new_line.replace("#.#","#")
            new_line = new_line.replace("#,#", '#')
            new_line = new_line.replace("#.#","#")    
        new_line = new_line.strip()
        new_line = " ".join(new_line.split())#remove useless whitespaces   
        print(new_line + '\n', file = treated_article)

