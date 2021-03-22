
import tensorflow as tf
from keras import backend as K
"""python file to import if one wants to use weighted binary cross entropy."""

def weighted_binary_crossentropy(pos_weight=0.78/0.22):
    def _to_tensor(x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)
  
  
    def _calculate_weighted_binary_crossentropy(target, output, from_logits=False):
        """Calculate weighted binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = _to_tensor(K.common.epsilon(), output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.math.log(output / (1 - output))
        return tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=pos_weight)


    def _weighted_binary_crossentropy(y_true, y_pred):
        return tf.math.reduce_mean(_calculate_weighted_binary_crossentropy(tf.cast(y_true, tf.float32), y_pred), axis=-1)
    
    return _weighted_binary_crossentropy