'''
4.5.2.2 早期終了 - Keras (MNIST)
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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

    x_train, x_val, t_train, t_val = \
        train_test_split(x_train, t_train, test_size=0.2)

    '''
    2. モデルの構築
    '''
    model = Sequential()
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',
                       patience=5,
                       verbose=1)

    hist = model.fit(x_train, t_train,
                     epochs=1000, batch_size=100,
                     verbose=2,
                     validation_data=(x_val, t_val),
                     callbacks=[es])

    '''
    4. モデルの評価
    '''
    # 誤差の可視化
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    fig = plt.figure()
    plt.rc('font', family='serif')
    plt.plot(range(len(loss)), loss,
             color='gray', linewidth=1,
             label='loss')
    plt.plot(range(len(val_loss)), val_loss,
             color='black', linewidth=1,
             label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.savefig('output.jpg')
    plt.show()

    # テストデータの評価
    loss, acc = model.evaluate(x_test, t_test, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        acc
    ))
