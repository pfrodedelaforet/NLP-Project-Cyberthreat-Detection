"""
Implement a PrototypeProjection callback
"""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from itertools import chain, islice
import numpy as np
from tqdm import tqdm
class PrototypeProjection(Callback):
    """Implements the "Prototype Projection" step during training.

    The associated `model` must have a sub-model called `encoder`
    and another called `prototypes_layer` (which gets its weights projected)

    Parameters
    ----------
    train_gen : prosenet.DataGenerator
        A generator for data from which to compute encodings --> project
    freq : int, optional
        How often to execute the projection, in epochs, default=4.
    print_argmins : bool, optional
        If True, print indices of closest matches in `train_gen`, default=False."""
    
    def __init__(self, train_dat, freq=4, print_argmins=False, **kwargs):
        self.train_dat = train_dat
        self.freq = freq
        if print_argmins:
            # need to verify behavior of `train_gen`
            raise NotImplementedError()
        super(PrototypeProjection, self).__init__(**kwargs)


    def on_epoch_end(self, epoch, logs=None):
        print(epoch)
        """
        Prototype projection computation + setting
        """
        if epoch % self.freq == 0:
            print('\nComputing prototype projection...')
            protos = self.model.prototypes_layer.weights[0][0]#(30, 768)
            # print(f'protos:{tf.shape(protos)}')
            # get encodings of all train sequences
            path = '/home/pfrod/storage/liste_encoded_precomputed'
            for i, x in tqdm(enumerate(self.train_dat._input_dataset.unbatch().batch(1))) : #ca fait seulement un repeat
                x_enc = self.model.encoder(x)[1]
                if epoch == 0 : 
                    np.save(path+f'/liste_encoded_precomputed{i}', x_enc.numpy())
                
                X_encoded = tf.convert_to_tensor(np.load(path+f'/liste_encoded_precomputed{i}.npy'))

                #print(X_encoded)
                # distance matrix from protos
                if i == 0 : 
                    d2 = tf.expand_dims(-tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(X_encoded, axis = -1), tf.nn.l2_normalize(protos, axis = -1)), axis=-1), 0)
        
                    
                else : 
                    new = tf.expand_dims(-tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(X_encoded, axis = -1), tf.nn.l2_normalize(protos, axis = -1)), axis=-1), 0)
                    d2 = tf.concat([d2, new], 0)
            idxs = tf.argmin(d2, axis=0)
            print(f'idxs:{idxs}')
            X_encoded = []
            for i in idxs : 
                X_encoded.append(tf.convert_to_tensor(np.load(path+f'/liste_encoded_precomputed{i}.npy')))
            new_protos = tf.concat(X_encoded, axis = 0)#(30, 768)
            # reset protos to nearest neighbors  
            new_protos = tf.reshape(new_protos, (1, *protos.shape)) # need to swap axes
            print(f'shapes : {tf.shape(new_protos)}, {tf.shape(self.model.prototypes_layer.weights)}')#(1, 15, 768), (1, 1, 15, 768)
            self.model.prototypes_layer.weights[0].assign(new_protos)
            print('... assigned new prototypes from projections.')
            f = open('/home/pfrod/results/nn_indices_prosenet.txt', 'a+')
            print(f'15 nearest neighbour text indices, epoch {epoch} : {list(zip(idxs.numpy(), tf.linalg.diag_part(tf.gather(d2, idxs)).numpy()))}', file = f)
            f.close()