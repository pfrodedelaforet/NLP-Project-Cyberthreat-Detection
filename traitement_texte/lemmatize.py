import re
from nltk.stem.wordnet import WordNetLemmatizer
import fileinput
from pathlib import Path
PATH = Path("../storage/dataset/lemmatized")
lemmatizer = WordNetLemmatizer()
for label in PATH.iterdir():
    for file in label.iterdir():
        for line in fileinput.input(str(file), inplace=True, backup='.bak'):
            line = ' '.join(
                [lemmatizer.lemmatize(w) for w in line.rstrip().split()]
            )
            # overwrites current `line` in file
            print(line)