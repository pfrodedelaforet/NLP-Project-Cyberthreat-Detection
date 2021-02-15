import torch
import linecache 
from transformers import LongformerTokenizer
tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 2048)
class Dataset_alamano(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, text, label):
        'Initialization'
        self.label = torch.LongTensor(label)
        self.text = text

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.text)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.text[index]
        y = self.label[index]
        return X, y
    

class Dataset_unlabelled_by_line(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, filename):
        'Initialization'
        self.file = filename
        self.len = sum(1 for line in open(filename, 'r', encoding='utf-8'))

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = tokenizer(linecache.getline(self.file, index))
        
        return X
