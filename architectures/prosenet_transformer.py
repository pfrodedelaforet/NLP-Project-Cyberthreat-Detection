from transformers import TFBertForSequenceClassification, BertTokenizer, TFTrainer, TFTrainingArguments, TFRobertaForSequenceClassification, RobertaTokenizer, TFRobertaModel, AdamWeightDecay, TFLongformerModel, LongformerTokenizer
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import sys
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
sys.path.insert(1, '/home/pfrod/architectures/tf-ProSeNet/prosenet')
#from prototypes import Prototypes
from prototypes_2 import Prototypes
# from projection import PrototypeProjection
from projection_2 import PrototypeProjection
from tensorflow.raw_ops import RepeatDataset
import tensorflow_addons as tfa
from operator import itemgetter
from bce_weights import weighted_binary_crossentropy
tf.keras.backend.clear_session()
gpu_act = True
if gpu_act : 
    GPU = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPU[0], True)
    #tf.config.experimental.set_virtual_device_configuration(GPU[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192//2)])
class InterpretableBertModel(tf.keras.Model):
    def __init__(self, k, list_dense_params, tokenizer_vocab_len, **kwargs):
        super(InterpretableBertModel, self).__init__(**kwargs)
        self.k = k
        self.list_dense_params = list_dense_params
        #self.encoder = TFRobertaModel.from_pretrained('roberta-base')
        self.encoder = TFLongformerModel.from_pretrained('longformer_128', from_pt = True, attention_window = 128, trainable = False)
        self.encoder.resize_token_embeddings(tokenizer_vocab_len)
        self.prototypes_layer = Prototypes(k=self.k)
        self.list_dense = [Dense(units=param[0], activation=param[1]) for param in list_dense_params]
        self.sigmoid = Dense(units=1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.similarity_vector(inputs)
        for layer in self.list_dense : 
            x = layer(x)
            x = Dropout(0.1)(x)
        return self.sigmoid(x)
    
    def similarity_vector(self, x):
        """Return the similarity vector(s) of shape (batches, k,)."""
        r_x = self.encoder(x)[1]
        return self.prototypes_layer(r_x)

#tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path = '../storage/tokenizer_roberta', max_length = 256, trainable = True)
tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 2048)
k = 15
list_dense_params = [(32, 'gelu'), (32, 'gelu')]#, (32, 'gelu')
global_model = InterpretableBertModel(k=k, list_dense_params=list_dense_params, tokenizer_vocab_len = len(tokenizer))

PATH = Path("../storage/treated_articles")

iterd = PATH.iterdir()
dat = []
labels = []
for label in iterd:
    for article in tqdm(label.iterdir()):
        dat.append(str(article))
        lab = str(label)[-17 :] == '/RELEVANT_TREATED'
        
        labels.append(lab)
        
files_train, files_test, y_train, y_test = train_test_split(dat, labels, test_size = 0.33, shuffle = True)
files_test, y_test = files_test, y_test
f = open('/home/pfrod/results/num_shuffled_articles.txt', 'a')
for i, (file, y) in enumerate(zip(files_train, y_train)):
    print(f'{i}'+', '+f'{file}'+', '+f'{y}', file = f)
f.close()

def clean_read(filename):
    return open(filename, 'r').read()\
            .replace('\n\n','. ')\
            .replace('..', '.')\
            .replace('\n', '')
def create_data_set(files, labels):
    inputs = tokenizer([clean_read(x) for x in files], return_tensors="tf", padding='max_length', max_length=2048, truncation=True)
    inputs = {k: inputs[k] for k in ['input_ids', 'attention_mask']}
    labels = tf.reshape(tf.constant(labels, dtype=tf.int32), (len(labels), 1))
    d_inputs = Dataset.from_tensor_slices(inputs)
    d_labels = Dataset.from_tensor_slices(labels)
    return Dataset.zip((d_inputs, d_labels))
##define datasets
data_train = create_data_set(files_train, y_train).batch(6).repeat(10)
data_test = create_data_set(files_test, y_test).batch(6).repeat(10)
#define callbacks
projection = PrototypeProjection(data_train, freq = 1)
log_dir = '../results/logging_prosenet_bert'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
##fit

global_model.compile(optimizer = AdamWeightDecay(learning_rate=1e-3), loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])#weighted_binary_crossentropy()
global_model.fit(data_train, epochs = 2, steps_per_epoch = 700, validation_data=data_test, callbacks=[projection, tensorboard_callback], shuffle=False, validation_steps = 101)#projection

global_model.save_weights('interpretablebert/model')
global_model.evaluate(data_test, verbose = 2)
new_model = InterpretableBertModel(k=k, list_dense_params=list_dense_params, tokenizer_vocab_len = len(tokenizer))
new_model.compile(optimizer = AdamWeightDecay(1e-3), loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
new_model.load_weights('interpretablebert/model')
new_model.evaluate(data_test, verbose = 2)

