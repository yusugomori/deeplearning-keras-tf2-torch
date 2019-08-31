'''
5.2.7 LSTM - TensorFlow (Adding Problem)
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


class RNN(Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.l1 = LSTM(hidden_dim, activation='tanh',
                       recurrent_activation='sigmoid',
                       kernel_initializer='glorot_normal',
                       recurrent_initializer='orthogonal')
        self.l2 = Dense(1, activation='linear')

    def call(self, x):
        h = self.l1(x)
        y = self.l2(h)

        return y


if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. データの準備
    '''
    def mask(T=200):
        mask = np.zeros(T)
        indices = np.random.permutation(np.arange(T))[:2]
        mask[indices] = 1
        return mask

    def toy_problem(N, T=200):
        signals = np.random.uniform(low=0.0, high=1.0,
                                    size=(N, T))
        masks = np.zeros((N, T))
        for i in range(N):
            masks[i] = mask(T)

        data = np.zeros((N, T, 2))
        data[:, :, 0] = signals[:]
        data[:, :, 1] = masks[:]
        target = (signals * masks).sum(axis=1).reshape(N, 1)

        return (data.astype(np.float32),
                target.astype(np.float32))

    N = 10000
    T = 200
    maxlen = T

    x, t = toy_problem(N, T)
    x_train, x_val, t_train, t_val = \
        train_test_split(x, t, test_size=0.2, shuffle=False)

    '''
    2. モデルの構築
    '''
    model = RNN(50)

    '''
    3. モデルの学習
    '''
    criterion = losses.MeanSquaredError()
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)
    train_loss = metrics.Mean()
    val_loss = metrics.Mean()

    def compute_loss(t, y):
        return criterion(t, y)

    def train_step(x, t):
        with tf.GradientTape() as tape:
            preds = model(x)
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return loss

    def val_step(x, t):
        preds = model(x)
        loss = compute_loss(t, preds)
        val_loss(loss)

    epochs = 500
    batch_size = 100
    n_batches_train = x_train.shape[0] // batch_size
    n_batches_val = x_val.shape[0] // batch_size
    hist = {'loss': [], 'val_loss': []}

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

        hist['loss'].append(train_loss.result())
        hist['val_loss'].append(val_loss.result())

        print('epoch: {}, loss: {:.3}, val_loss: {:.3f}'.format(
            epoch+1,
            train_loss.result(),
            val_loss.result()
        ))

    '''
    4. モデルの評価
    '''
    # 誤差の可視化
    val_loss = hist['val_loss']

    fig = plt.figure()
    plt.rc('font', family='serif')
    plt.plot(range(len(val_loss)), val_loss,
             color='black', linewidth=1,
             label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.savefig('output.jpg')
    plt.show()
