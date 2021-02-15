import numpy as np
import tensorflow as tf 
from tensorflow import keras 

"""Attempt, in order to deeply understand how a transformer works, to recode an encoding layer by myself."""
class MultiHeadAttentionLayer(keras.layers.Layer):
    def __init__(self, n_heads, units):
        super(MultiHeadAttentionLayer, self).__init__()
        self.list_heads = []
        self.units = units
        self.n_heads = n_heads
    def build(self, input_shape):
        n_output = input_shape[-1]
        for _ in range(self.n_heads) : 
            W_Q = tf.keras.layers.Dense(self.units, activation = 'tanh')
            W_K = tf.keras.layers.Dense(self.units, activation = 'tanh')
            W_V = tf.keras.layers.Dense(self.units, activation = 'tanh')
            self.list_heads.append([W_Q, W_K, W_V])
        self.W_O = tf.keras.layers.Dense(n_output, activation = 'linear')
    def call(self, inputs) : 
        list_attentions = []
        batch_size, n_words_window = inputs.shape[:-1]
        inp = tf.reshape(inputs, [batch_size * n_words_window, inputs.shape[2]])
        for x in self.list_heads : 
            q = tf.reshape(x[0](inp), [batch_size, n_words_window, self.units])
            K = tf.reshape(x[1](inp), [batch_size, n_words_window, self.units])
            V = tf.reshape(x[2](inp), [batch_size, n_words_window, self.units])
            soft = tf.nn.softmax(tf.matmul(q, tf.transpose(K, perm = [0, 2, 1]))/8, axis = 2)
            list_attentions.append(tf.matmul(soft, V))#là on devrait avoir la même shape que q, k, v, soit (b, w, units) pour chaque 
        print(list_attentions[0].shape)
        z_res = tf.concat(list_attentions, 2)# (b, w, units * n_heads)
        print(f'z_res : {z_res.shape}')
        return self.W_O(z_res)


def positionnal_encoding(pos, embedding_size):
    PE = np.zeros(pos.shape+(embedding_size,))
    for i in range(embedding_size):
        if i % 2 == 0:
            PE[:, :, i] = np.sin(pos / 10000 ** (i / embedding_size))
        else:
            PE[:, :, i] = np.cos(pos / 10000 ** ((i - 1) / embedding_size))
    return PE


class Encoder(keras.layers.Layer):
    def __init__(self, n_heads, output_activation, list_dense, units = 10):
        super(Encoder, self).__init__()
        self.n_heads = n_heads
        self.units = units
        self.list_dense = list_dense
        self.output_activation = output_activation

    def build(self, input_shape):
        self.mhal = MultiHeadAttentionLayer(n_heads = self.n_heads, units = self.units)
        self.dense_layers = [tf.keras.layers.Dense(x, activation = y) for x, y in self.list_dense]
        self.output_layer = tf.keras.layers.Dense(input_shape[-1], activation = self.output_activation)
    def call(self, inputs):
        batch_size, n_words_window = inputs.shape[:-1]
        x = self.mhal(inputs)
        x = tf.add(x, inputs)
        interm = tf.linalg.normalize(x, ord = 1, axis = 1)[0]
        print(x.shape[0], x.shape[1], batch_size, n_words_window)
        x = tf.reshape(interm, [batch_size * n_words_window, x.shape[2]])
        for layer in self.dense_layers : 
            x = layer(x)
        x = self.output_layer(x)
        x = tf.add(interm, tf.reshape(x, [batch_size, n_words_window, x.shape[1]]))
        x = tf.linalg.normalize(x, ord = 1, axis = 1)[0]#normalement shape = (b,w,embedding)
        return x

input_dims = (4, 7)
#inp = Input(())
voc = list(set(list('ma bicht aime la quiche')))+["ma", "bi", "cht", "ai", "me", "la", "qu", "che"]
#print(f"voc : {len(voc)}")
batch_size = 10
n_words_window = 30 # à régler
embedding_size = 300
n_heads = 10
encoder_list_dense = [(32, 'relu')]
encoder_output_activation = 'relu'
if __name__ == "__main__":
    inp = keras.Input((n_words_window,))
    x = tf.keras.layers.Embedding(output_dim = embedding_size, input_length = n_words_window, input_dim = len(voc))(inp)
    PE = tf.constant(positionnal_encoding(np.vstack([np.arange(n_words_window)]*batch_size), embedding_size), dtype='float32')
    x = tf.add(x, PE)
    encoder = Encoder(n_heads = n_heads, output_activation = encoder_output_activation, list_dense = encoder_list_dense)
    x = encoder(x)
    x = tf.reduce_sum(x, axis = 1)
    x = tf.keras.layers.Dense(32, activation = 'linear')(x)
    out = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    model.fit(np.random.randint(len(voc), size = (30, 30)), np.random.randint(2, size = 30), batch_size = batch_size, verbose = True)
    print(out.shape)
