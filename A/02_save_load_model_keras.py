'''
A.3.1 モデルの保存と読み込み - Keras
'''

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import \
    Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model


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
    model.add(Dense(200, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_initializer='he_normal',
                    activation='softmax'))

    '''
    3. モデルの学習・保存
    '''
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, t_train,
              epochs=10, batch_size=100,
              verbose=2,
              validation_data=(x_val, t_val))

    model.save('model_keras.h5')  # モデルを保存

    print('model saved to: {}'.format('model_keras.h5'))

    '''
    4. モデルの読み込み・評価
    '''
    del model  # これまで学習していたモデルを削除

    model = load_model('model_keras.h5')  # 学習済のモデルを読み込み

    print('-' * 20)
    print('model loaded.')

    # テストデータの評価
    loss, acc = model.evaluate(x_test, t_test, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        acc
    ))
