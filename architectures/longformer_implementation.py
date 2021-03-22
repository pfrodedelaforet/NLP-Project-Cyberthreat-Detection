import datasets
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, RobertaTokenizer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import itertools
from torch.utils.data import DataLoader
from torch.utils import tensorboard
#from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
from dataset_pytorch import Dataset_alamano
from DataCollator import DataCollator
from compute_metrics import compute_metrics
"""Implementation of the Longformer, used before dealing with ProSeNet"""
PATH = Path("../storage/treated_articles")
iterd = PATH.iterdir()
dat = []

labels = []
for label in iterd:#on peut bien modifier iterdir c'est pas grave
    for article in tqdm(label.iterdir()):
        dat.append(str(article))
        labels.append(str(label)[-17 :] == '/RELEVANT_TREATED')
#print(dat[:100])
files_train, files_test, y_train, y_test = train_test_split(dat, labels, test_size = 0.33, shuffle = True)
def generator(dirs):
    def gen():
        for file in dirs :
            yield open(file, 'r').read()
    return gen
#x_train =  Dataset.from_generator(generator(files_train), output_types = 'string')
x_train = [open(file, 'r').read().replace('\n\n','. ').replace('..', '.').replace('\n', '') for file in files_train]
x_test = [open(file, 'r').read().replace('\n\n','. ').replace('..', '.').replace('\n', '') for file in files_test]
data_train = Dataset_alamano(x_train, y_train)
data_test = Dataset_alamano(x_test, y_test)
i = 0

model = LongformerForSequenceClassification.from_pretrained('pretrained_finetune',
                                                           gradient_checkpointing=True,
                                                           attention_window = 128)
                
model.to(torch.device('cuda:0'))
tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 2048)
#print(tokenizer('piche', padding = 'max_length', truncation = True))
def tokenization(batched_text):
    return tokenizer(batched_text, padding = 'max_length', truncation=True, max_length = 2048)

training_args = TrainingArguments(
    output_dir = '../results/longformer_128',
    num_train_epochs = 8,#5
    per_device_train_batch_size = 8,#8
    gradient_accumulation_steps = 8,    
    per_device_eval_batch_size= 8,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    #load_best_model_at_end=True,
    warmup_steps=150,
    weight_decay=0.01,
    logging_steps = 4,
    fp16 = True,
    logging_dir='../results/logging_longformer_128',
    #dataloader_num_workers = 0,
    run_name = 'longformer-classification-updated-rtx3090_paper_replication_2_warm', 
)
writer = tensorboard.SummaryWriter(training_args.logging_dir)

trainer = Trainer(model = model, 
                  args = training_args, 
                  train_dataset = data_train, 
                  eval_dataset = data_test,
                  data_collator = DataCollator(tokenizer = tokenizer), 
                  compute_metrics = compute_metrics)
trainer.train()
trainer.save_model('longformer_128')
trainer.evaluate()