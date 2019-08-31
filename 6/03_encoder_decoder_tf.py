'''
6.1.4 Encoder-Decoder - TensorFlow
'''

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from utils import Vocab
from utils.tf import DataLoader


class EncoderDecoder(Model):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 maxlen=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

        self.maxlen = maxlen
        self.output_dim = output_dim

    def call(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.shape[0]
        if target is not None:
            len_target_sequences = target.shape[1]
        else:
            len_target_sequences = self.maxlen

        _, states = self.encoder(source)

        y = tf.ones((batch_size, 1), dtype=tf.int32)
        output = tf.zeros((batch_size, 1, self.output_dim), dtype=tf.float32)

        for t in range(len_target_sequences):
            out, states = self.decoder(y, states)
            out = out[:, tf.newaxis]
            output = tf.concat([output, out], axis=1)

            if use_teacher_forcing and target is not None:
                y = target[:, t][:, tf.newaxis]
            else:
                y = tf.argmax(out, axis=-1, output_type=tf.int32)

        return output[:, 1:]


class Encoder(Model):
    def __init__(self,
                 input_dim,
                 hidden_dim):
        super().__init__()
        self.embedding = Embedding(input_dim, hidden_dim, mask_zero=True)
        self.lstm = LSTM(hidden_dim, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer='glorot_normal',
                         recurrent_initializer='orthogonal',
                         return_state=True)

    def call(self, x):
        x = self.embedding(x)
        h, state_h, state_c = self.lstm(x)

        return h, (state_h, state_c)


class Decoder(Model):
    def __init__(self,
                 hidden_dim,
                 output_dim):
        super().__init__()
        self.embedding = Embedding(output_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer='glorot_normal',
                         recurrent_initializer='orthogonal',
                         return_state=True)
        self.out = Dense(output_dim, kernel_initializer='glorot_normal',
                         activation='softmax')

    def call(self, x, states):
        x = self.embedding(x)
        h, state_h, state_c = self.lstm(x, states)
        y = self.out(h)

        return y, (state_h, state_c)


if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. データの準備
    '''
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    en_train_path = os.path.join(data_dir, 'train.en')
    en_val_path = os.path.join(data_dir, 'dev.en')
    en_test_path = os.path.join(data_dir, 'test.en')

    ja_train_path = os.path.join(data_dir, 'train.ja')
    ja_val_path = os.path.join(data_dir, 'dev.ja')
    ja_test_path = os.path.join(data_dir, 'test.ja')

    en_vocab = Vocab()
    ja_vocab = Vocab()

    en_vocab.fit(en_train_path)
    ja_vocab.fit(ja_train_path)

    x_train = en_vocab.transform(en_train_path)
    x_val = en_vocab.transform(en_val_path)
    x_test = en_vocab.transform(en_test_path)

    t_train = ja_vocab.transform(ja_train_path, eos=True)
    t_val = ja_vocab.transform(ja_val_path, eos=True)
    t_test = ja_vocab.transform(ja_test_path, eos=True)

    def sort(x, t):
        lens = [len(i) for i in x]
        indices = sorted(range(len(lens)), key=lambda i: -lens[i])
        x = [x[i] for i in indices]
        t = [t[i] for i in indices]

        return (x, t)

    (x_train, t_train) = sort(x_train, t_train)
    (x_val, t_val) = sort(x_val, t_val)
    (x_test, t_test) = sort(x_test, t_test)

    train_dataloader = DataLoader((x_train, t_train))
    val_dataloader = DataLoader((x_val, t_val))
    test_dataloader = DataLoader((x_test, t_test), batch_size=1)

    '''
    2. モデルの構築
    '''
    depth_x = len(en_vocab.i2w)
    depth_t = len(ja_vocab.i2w)

    input_dim = depth_x
    hidden_dim = 128
    output_dim = depth_t

    model = EncoderDecoder(input_dim, hidden_dim, output_dim)

    '''
    3. モデルの学習・評価
    '''
    criterion = tf.losses.CategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)
    train_loss = metrics.Mean()
    val_loss = metrics.Mean()

    def compute_loss(t, y):
        return criterion(t, y)

    def train_step(x, t, depth_t,
                   teacher_forcing_rate=0.5):
        use_teacher_forcing = (random.random() < teacher_forcing_rate)
        with tf.GradientTape() as tape:
            preds = model(x, t, use_teacher_forcing=use_teacher_forcing)
            mask_t = tf.cast(tf.not_equal(t, 0), tf.float32)
            t = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
            t = t * mask_t[:, :, tf.newaxis]
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return preds

    def val_step(x, t, depth_t):
        preds = model(x, t, use_teacher_forcing=False)
        mask_t = tf.cast(tf.not_equal(t, 0), tf.float32)
        t = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
        t = t * mask_t[:, :, tf.newaxis]
        loss = compute_loss(t, preds)
        val_loss(loss)

        return preds

    def test_step(x):
        preds = model(x)
        return preds

    epochs = 30

    for epoch in range(epochs):
        print('-' * 20)
        print('epoch: {}'.format(epoch+1))

        for (x, t) in train_dataloader:
            train_step(x, t, depth_t)

        for (x, t) in val_dataloader:
            val_step(x, t, depth_t)

        print('loss: {:.3f}, val_loss: {:.3}'.format(
            train_loss.result(),
            val_loss.result()
        ))

        for idx, (x, t) in enumerate(test_dataloader):
            preds = test_step(x)

            source = x.numpy().reshape(-1)
            target = t.numpy().reshape(-1)
            out = tf.argmax(preds, axis=-1).numpy().reshape(-1)

            source = ' '.join(en_vocab.decode(source))
            target = ' '.join(ja_vocab.decode(target))
            out = ' '.join(ja_vocab.decode(out))

            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 9:
                break
