'''
6.3.4.4 Positional Encoding - TensorFlow
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    def __init__(self, output_dim,
                 maxlen=6000):
        super().__init__()
        self.output_dim = output_dim
        self.maxlen = maxlen

    def build(self, input_shape):
        self.PE = self.add_weight(name='PE',
                                  shape=(self.maxlen,
                                         self.output_dim),
                                  initializer=self.initializer,
                                  trainable=False,
                                  dtype=tf.float32)

        super().build(input_shape)

    def call(self, x):
        pe = self.PE[tf.newaxis, :tf.shape(x)[1], :]
        return x + pe

    def initializer(self, input_shape, dtype=tf.float32):
        pe = \
            np.array([[pos / np.power(10000, 2 * (i // 2) / self.output_dim)
                       for i in range(self.output_dim)]
                      for pos in range(self.maxlen)])

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        return tf.convert_to_tensor(pe, dtype=tf.float32)
