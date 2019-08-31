'''
5.3.2 GRU - Keras (Adding Problem)
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import pickle


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
    model = Sequential()
    model.add(GRU(50, activation='tanh',
                  recurrent_activation='sigmoid',
                  kernel_initializer='glorot_normal',
                  recurrent_initializer='orthogonal'))
    model.add(Dense(1, activation='linear'))

    '''
    3. モデルの学習
    '''
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    hist = model.fit(x_train, t_train,
                     epochs=500, batch_size=100,
                     verbose=2,
                     validation_data=(x_val, t_val))

    '''
    4. モデルの評価
    '''
    # 誤差の可視化
    val_loss = hist.history['val_loss']

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
