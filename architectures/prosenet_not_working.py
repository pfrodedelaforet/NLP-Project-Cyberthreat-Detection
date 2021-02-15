"""from pytorch2keras.converter import pytorch_to_keras
from transformers import TFLongformerForSequenceClassification
model = TFLongformerForSequenceClassification.from_pretrained('longformer_wout_finetune')
"""
"""
import onnx
import tensorflow as tf
from tensorflow.keras.models import load_model
onnx.convert('longformer_wout_finetune', 'tf_longformer_wout_finetune.hdf5')
model = load_model('tf_longformer_wout_finetune.hdf5')
model.predict(['I go to school'])
"""
"""
from transformers import TFLongformerForSequenceClassification, LongformerTokenizer, TFTrainer, TFTrainingArguments, LongformerForSequenceClassification, LongformerConfig, TFLongformerModel
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#from prototypes import Prototypes
#tf.keras.backend.set_floatx('float16')
gpu_act = True
if gpu_act : 
    GPU = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPU[0], True)
    tf.config.experimental.set_virtual_device_configuration(GPU[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000//2)])

tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 2048)


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

def gen(files, labels):
    for file, label in zip(files_train, y_train):
        inputs = tokenizer.encode_plus(open(file, 'r').read().replace('\n\n','. ').replace('..', '.').replace('\n', ''), padding = 'max_length', truncation = True, max_length = 2048, return_tensors = 'tf')
        #inputs['labels'] = label
        yield inputs, tf.convert_to_tensor(label, dtype = tf.int32)

#data_train = Dataset.from_generator(gen, args = (files_train, y_train), output_types = ({'input_ids':tf.int32, 'attention_mask':tf.int32}, tf.int32))
#data_train = Dataset.from_tensor_slices({'input_ids' : component_gen('input_ids'), 'attention mask': component_gen('attention_mask'), 'labels' : component_gen('labels')}).batch(8)
#remplacer dictionnaire par une liste 
#data_test = Dataset.from_generator(gen, args = (files_test, y_test), output_types = ({'input_ids':tf.int32, 'attention_mask':tf.int32}, tf.int32))
#data_train = Dataset.from_tensor_slices({'input_ids' : component_gen('input_ids', True), 'attention mask': component_gen('attention_mask', True), 'labels' : component_gen('labels', True)}).batch(8)
#ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#data_train = gen(files_train, y_train)
x_t = {'input_ids' : [None]*len(files_train), 'attention_mask' : [None]*len(files_train)}
for i, file in enumerate(files_train) : 
    tok = tokenizer(open(file, 'r').read().replace('\n\n','. ').replace('..', '.').replace('\n', ''), padding = 'max_length', truncation = True, max_length = 2048, return_tensors = 'tf')
    x_t['input_ids'][i] = tok['input_ids'][0]
    x_t['attention_mask'][i] = tok['attention_mask'][0]

x_te = {'input_ids' : [None]*len(files_test), 'attention_mask' : [None]*len(files_test)}
for i, file in enumerate(files_test) : 
    tok = tokenizer(open(file, 'r').read().replace('\n\n','. ').replace('..', '.').replace('\n', ''), padding = 'max_length', truncation = True, max_length = 2048, return_tensors = 'tf')
    x_te['input_ids'][i] = tok['input_ids'][0]
    x_te['attention_mask'][i] = tok['attention_mask'][0]
x_t['input_ids'] = tf.convert_to_tensor(x_t['input_ids'], dtype=tf.int32)
x_t['attention_mask'] = tf.convert_to_tensor(x_t['attention_mask'], dtype=tf.int32)
x_te['input_ids'] = tf.convert_to_tensor(x_te['input_ids'], dtype=tf.int16)
x_te['attention_mask'] = tf.convert_to_tensor(x_te['attention_mask'], dtype=tf.int32)
#x_t = {'input_ids' : tf.convert_to_tensor(np.vstack([np.random.randint(1000, size = 2048) for _ in range(5)])), 'attention_mask' : tf.convert_to_tensor(np.vstack([np.random.randint(2, size = 2048) for _ in range(5)])), 'labels' : tf.convert_to_tensor([True, False, True, True, False])}
#print(model(dict_test))
#y_t = tf.convert_to_tensor([True, False, True, True, False])
#dict : {'input_ids' : [[1, 2, 6], [2, 3, 900]], 'attention_mask' : [[0, 1, 0]], 'ids' : [1, 2, 3, 4, ]}
data_x_train = Dataset.from_tensor_slices(x_t)
data_y_train = Dataset.from_tensor_slices(list(map(int, y_train)))
data_train = Dataset.zip((data_x_train, data_y_train))#.batch(8).repeat(2)

data_x_test = Dataset.from_tensor_slices(x_te)
data_y_test = Dataset.from_tensor_slices(list(map(int, y_test)))
data_test = Dataset.zip((data_x_test, data_y_test))#.batch(8).repeat(2)
training_args = TFTrainingArguments(
    output_dir = '../results/interpretable_longformer',
    num_train_epochs = 8,#5
    per_device_train_batch_size = 2,#8
    gradient_accumulation_steps = 8,    
    per_device_eval_batch_size= 2,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    #load_best_model_at_end=True,
    warmup_steps=150,
    weight_decay=0.01,
    logging_steps = 4,
    fp16 = False,
    logging_dir='../results/logging_interpretable_longformer',
    #dataloader_num_workers = 1,
    run_name = 'longformer-classification-updated-rtx3090_paper_replication_2_warm', 
)

with training_args.strategy.scope():
        model = TFLongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                               gradient_checkpointing=True,
                                                               attention_window = 512, return_dict = True)
        
        model.resize_token_embeddings(len(tokenizer))
#model.compile(tf.keras.optimizers.Adam(learning_rate=3e-5), loss = tf.keras.losses.BinaryCrossentropy())#from_logits=True
#model.fit(data_train, epochs = 2, steps_per_epoch=115)


trainer = TFTrainer(model=model, args=training_args,
                               train_dataset=data_train, eval_dataset=data_test)

trainer.train()"""
"""
from transformers import BertTokenizer, glue_convert_examples_to_features
import tensorflow as tf
import tensorflow_datasets as tfds
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = tfds.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=2048, task='mrpc')
data_train = train_dataset.shuffle(100).batch(8)
test_dataset = glue_convert_examples_to_features(data['test'], tokenizer, max_length=2048, task='mrpc')
data_test = train_dataset.shuffle(100).batch(8)"""
"""
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5), loss = 'binary_crossentropy')
model.fit(data_train, epochs=3)"""

import os

from transformers import TFLongformerForSequenceClassification, LongformerTokenizer, TFTrainer, TFTrainingArguments
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
gpu_act = True
if gpu_act :
    try : 
        GPU = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(GPU[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
    except : 
        pass
# Load data
files_irrelevant = []
for root, dirs, files in os.walk("../storage/treated_articles/IRRELEVANT_TREATED"):
    for f in files:
        files_irrelevant.append(os.path.join(root, f))

files_relevant = []
for root, dirs, files in os.walk("../storage/treated_articles/RELEVANT_TREATED"):
    for f in files:
        files_relevant.append(os.path.join(root, f))

files = files_irrelevant + files_relevant
labels = [0]*len(files_irrelevant) + [1]*len(files_relevant)

tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 2048)

files_train, files_test, labels_train, labels_test = \
        train_test_split(files, labels, test_size = 0.33, shuffle = True)

def clean_read(filename):
    return open(filename, 'r').read()\
            .replace('\n\n','. ')\
            .replace('..', '.')\
            .replace('\n', '')

# Generate batch of slices_length
def gen(files, labels, slices_length = 1):
    for i in range(0, min(len(files),len(labels)), slices_length):
        f = files[i:i+slices_length]
        l = labels[i:i+slices_length]
        inputs = tokenizer([clean_read(x) for x in f], return_tensors="tf", padding='max_length', max_length=2048, truncation=True)
        print(inputs)
        inputs = {k: inputs[k] for k in ['input_ids', 'attention_mask']}
        inputs["labels"] = tf.reshape(tf.constant(l, dtype=tf.int32), (len(f), 1))
        yield inputs

def create_data_set(files, labels):
    inputs = tokenizer([clean_read(x) for x in files], return_tensors="tf", padding='max_length', max_length=2048, truncation=True)
    inputs = {k: inputs[k] for k in ['input_ids', 'attention_mask']}
    labels = tf.reshape(tf.constant(labels, dtype=tf.int32), (len(labels), 1))
    d_inputs = Dataset.from_tensor_slices(inputs)
    d_labels = Dataset.from_tensor_slices(labels)
    return Dataset.zip((d_inputs, d_labels))

data_train = create_data_set(files_train, labels_train)
data_test = create_data_set(files_test, labels_test)

training_args = TFTrainingArguments(
    output_dir = '../results/interpretable_longformer',
    num_train_epochs = 8,#5
    per_device_train_batch_size = 8,#8
    gradient_accumulation_steps = 8,
    #per_device_eval_batch_size= 8,
    evaluation_strategy = "epoch",
    disable_tqdm = False,
    #load_best_model_at_end=True,
    warmup_steps=150,
    weight_decay=0.01,
    logging_steps = 4,
    fp16 = False,
    logging_dir='../results/logging_interpretable_longformer',
    #dataloader_num_workers = 1,
    run_name = 'longformer-classification-updated-rtx3090_paper_replication_2_warm',
)

with training_args.strategy.scope():
    model = TFLongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                   gradient_checkpointing=True,
                                                                   attention_window = 512)

trainer = TFTrainer(model=model, args=training_args, train_dataset=data_train, eval_dataset=data_test)
trainer.train()
