import fasttext
from pathlib import Path
from tqdm import tqdm
"""This is the script removing the elements of the dataset in a language other than english. Normally everything is OK because we checked when labeling.
This is the very method used (just change a few lines) to see if each paragraph of the cybersecurity corpus was in english. If not, then the paragraph is removed."""
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