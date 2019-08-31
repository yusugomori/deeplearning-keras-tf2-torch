'''
6.1.3.3 データローダ - TensorFlow
'''

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets


class DataLoader(object):
    def __init__(self, dataset,
                 batch_size=100,
                 shuffle=False,
                 random_state=None):
        self.dataset = list(zip(dataset[0], dataset[1]))
        self.batch_size = batch_size
        self.shuffle = shuffle

        if random_state is None:
            random_state = np.random.RandomState(123)

        self.random_state = random_state
        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()

        x, t = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)

        self._idx += self.batch_size

        return x, t

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset,
                                   random_state=self.random_state)
        self._idx = 0


if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    mnist = datasets.mnist
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    t_train = np.eye(10)[t_train].astype(np.float32)

    train_dataloader = DataLoader((x_train, t_train),
                                  batch_size=100,
                                  shuffle=True)

    for (x, t) in train_dataloader:
        print(x.shape)  # => (100, 784)
        print(t.shape)  # => (100, 10)
        break
