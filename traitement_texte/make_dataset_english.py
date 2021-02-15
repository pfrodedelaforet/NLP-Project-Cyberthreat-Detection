import fasttext
from pathlib import Path
from tqdm import tqdm
model = fasttext.load_model('../lid.176.bin')
PATH = Path("../storage/dataset/articles")
list_stranger = []
for label in PATH.iterdir():
    for article in tqdm(label.iterdir()):
        with article.open('r', encoding = 'utf-8') as file :
            concat_file = ' '.join(file.readlines()).replace('\n', '')
            pred = model.predict(concat_file)[0][0] 
            if pred != '__label__en':
                print(concat_file)
                list_stranger.append(article)
print(list_stranger, len(list_stranger))