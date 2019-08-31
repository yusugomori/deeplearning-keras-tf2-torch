'''
5.4.4.2 BiRNN - TensorFlow (IMDb)
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers \
    import Dense, LSTM, Bidirectional, Embedding
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences


class BiRNN(Model):
    def __init__(self, num_words, hidden_dim):
        super().__init__()
        self.emb = Embedding(num_words, hidden_dim, mask_zero=True)
        self.lstm = Bidirectional(LSTM(hidden_dim, activation='tanh',
                                       recurrent_activation='sigmoid',
                                       kernel_initializer='glorot_normal',
                                       recurrent_initializer='orthogonal'),
                                  merge_mode='concat')
        self.out = Dense(1, kernel_initializer='glorot_normal',
                         activation='sigmoid')

    def call(self, x):
        h = self.emb(x)
        h = self.lstm(h)
        y = self.out(h)
        return tf.reshape(y, [-1])  # (batch_size, 1) => (batch_size,)


if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. データの準備
    '''
    num_words = 20000
    maxlen = 80

    imdb = datasets.imdb
    word_index = imdb.get_word_index()

    (x_train, t_train), (x_test, t_test) = imdb.load_data(num_words=num_words,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)

    x_train, x_val, t_train, t_val = \
        train_test_split(x_train, t_train, test_size=0.2)

    x_train = pad_sequences(x_train, maxlen=maxlen, padding='pre')
    x_val = pad_sequences(x_val, maxlen=maxlen, padding='pre')
    x_test = pad_sequences(x_test, maxlen=maxlen, padding='pre')

    '''
    2. モデルの構築
    '''
    model = BiRNN(num_words, 128)

    '''
    3. モデルの学習
    '''
    criterion = losses.BinaryCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)
    train_loss = metrics.Mean()
    train_acc = metrics.BinaryAccuracy()
    val_loss = metrics.Mean()
    val_acc = metrics.BinaryAccuracy()

    def compute_loss(t, y):
        return criterion(t, y)

    def train_step(x, t):
        with tf.GradientTape() as tape:
            preds = model(x)
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(t, preds)

        return loss

    def val_step(x, t):
        preds = model(x)
        loss = compute_loss(t, preds)
        val_loss(loss)
        val_acc(t, preds)

    epochs = 1000
    batch_size = 100
    n_batches_train = x_train.shape[0] // batch_size
    n_batches_val = x_val.shape[0] // batch_size
    es = EarlyStopping(patience=5, verbose=1)

    for epoch in range(epochs):
        x_, t_ = shuffle(x_train, t_train)

        for batch in range(n_batches_train):
            start = batch * batch_size
            end = start + batch_size
            train_step(x_[start:end], t_[start:end])

        for batch in range(n_batches_val):
            start = batch * batch_size
            end = start + batch_size
            val_step(x_val[start:end], t_val[start:end])

        print('epoch: {}, loss: {:.3}, acc: {:.3f}'
              ', val_loss: {:.3}, val_acc: {:.3f}'.format(
                  epoch+1,
                  train_loss.result(),
                  train_acc.result(),
                  val_loss.result(),
                  val_acc.result()
              ))

        if es(val_loss.result()):
            break

    '''
    4. モデルの評価
    '''
    test_loss = metrics.Mean()
    test_acc = metrics.BinaryAccuracy()

    def test_step(x, t):
        preds = model(x)
        loss = compute_loss(t, preds)
        test_loss(loss)
        test_acc(t, preds)

        return loss

    test_step(x_test, t_test)

    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        test_loss.result(),
        test_acc.result()
    ))
