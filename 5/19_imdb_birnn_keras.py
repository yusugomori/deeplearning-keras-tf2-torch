'''
5.4.4.2 BiRNN - Keras (IMDb)
'''

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers \
    import Dense, LSTM, Bidirectional, Embedding
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
    model = Sequential()
    model.add(Embedding(num_words, 128, mask_zero=True))
    model.add(Bidirectional(LSTM(128, activation='tanh',
                                 recurrent_activation='sigmoid',
                                 kernel_initializer='glorot_normal',
                                 recurrent_initializer='orthogonal'),
                            merge_mode='concat'))
    model.add(Dense(1, kernel_initializer='glorot_normal',
                    activation='sigmoid'))

    '''
    3. モデルの学習
    '''
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',
                       patience=5,
                       verbose=1)

    model.fit(x_train, t_train,
              epochs=1000, batch_size=100,
              verbose=2,
              validation_data=(x_val, t_val),
              callbacks=[es])

    '''
    4. モデルの評価
    '''
    test_loss, test_acc = model.evaluate(x_test, t_test, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        test_loss,
        test_acc
    ))
