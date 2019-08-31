'''
4.3.1.2 tanh - Keras (MNIST)
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. データの準備
    '''
    mnist = datasets.mnist
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)
    t_train = np.eye(10)[t_train].astype(np.float32)
    t_test = np.eye(10)[t_test].astype(np.float32)

    '''
    2. モデルの構築
    '''
    model = Sequential()
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, t_train,
              epochs=30, batch_size=100,
              verbose=2)

    '''
    4. モデルの評価
    '''
    loss, acc = model.evaluate(x_test, t_test, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        acc
    ))
