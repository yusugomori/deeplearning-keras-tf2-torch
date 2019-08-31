'''
6.3.4.5 Transformer - TensorFlow
'''

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from layers.tf import PositionalEncoding
from layers.tf import LayerNormalization
from layers.tf import MultiHeadAttention
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from utils import Vocab
from utils.tf import DataLoader


class Transformer(Model):
    def __init__(self,
                 depth_source,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.encoder = Encoder(depth_source,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen)
        self.decoder = Decoder(depth_target,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen)
        self.out = Dense(depth_target, activation='softmax')
        self.maxlen = maxlen

    def call(self, source, target=None):
        mask_source = self.sequence_mask(source)

        hs = self.encoder(source, mask=mask_source)

        if target is not None:
            target = target[:, :-1]
            len_target_sequences = target.shape[1]
            mask_target = self.sequence_mask(target)[:, tf.newaxis, :]
            subsequent_mask = self.subsequence_mask(target)
            mask_target = tf.greater(mask_target + subsequent_mask, 1)

            y = self.decoder(target, hs,
                             mask=mask_target,
                             mask_source=mask_source)
            output = self.out(y)
        else:
            batch_size = source.shape[0]
            len_target_sequences = self.maxlen

            output = tf.ones((batch_size, 1), dtype=tf.int32)

            for t in range(len_target_sequences - 1):
                mask_target = self.subsequence_mask(output)
                out = self.decoder(output, hs,
                                   mask=mask_target,
                                   mask_source=mask_source)
                out = self.out(out)[:, -1:, :]
                out = tf.argmax(out, axis=-1, output_type=tf.int32)
                output = tf.concat([output, out], axis=-1)

        return output

    def sequence_mask(self, x):
        len_sequences = \
            tf.reduce_sum(tf.cast(tf.not_equal(x, 0),
                                  tf.int32), axis=1)
        mask = \
            tf.cast(tf.sequence_mask(len_sequences,
                                     tf.shape(x)[-1]), tf.float32)
        return mask

    def subsequence_mask(self, x):
        shape = (x.shape[1], x.shape[1])
        mask = np.tril(np.ones(shape, dtype=np.int32), k=0)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        return tf.tile(mask[tf.newaxis, :, :], [x.shape[0], 1, 1])


class Encoder(Model):
    def __init__(self,
                 depth_source,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.embedding = Embedding(depth_source,
                                   d_model, mask_zero=True)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.encoder_layers = [
            EncoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen) for _ in range(N)
        ]

    def call(self, x, mask=None):
        x = self.embedding(x)
        y = self.pe(x)
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y, mask=mask)

        return y


class EncoderLayer(Model):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()

    def call(self, x, mask=None):
        h = self.attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y)

        return y


class Decoder(Model):
    def __init__(self,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.embedding = Embedding(depth_target,
                                   d_model, mask_zero=True)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.decoder_layers = [
            DecoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen) for _ in range(N)
        ]

    def call(self, x, hs,
             mask=None,
             mask_source=None):
        x = self.embedding(x)
        y = self.pe(x)

        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, hs,
                              mask=mask,
                              mask_source=mask_source)

        return y


class DecoderLayer(Model):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = Dropout(p_dropout)
        self.norm3 = LayerNormalization()

    def call(self, x, hs,
             mask=None,
             mask_source=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        z = self.src_tgt_attn(h, hs, hs,
                              mask=mask_source)
        z = self.dropout2(z)
        z = self.norm2(h + z)

        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)

        return y


class FFN(Model):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1 = Dense(d_ff, activation='relu')
        self.l2 = Dense(d_model, activation='linear')

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

    t_train = ja_vocab.transform(ja_train_path, bos=True, eos=True)
    t_val = ja_vocab.transform(ja_val_path, bos=True, eos=True)
    t_test = ja_vocab.transform(ja_test_path, bos=True, eos=True)

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

    model = Transformer(depth_x,
                        depth_t,
                        N=3,
                        h=4,
                        d_model=128,
                        d_ff=256,
                        maxlen=20)

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

    def train_step(x, t, depth_t):
        with tf.GradientTape() as tape:
            preds = model(x, t)
            t = t[:, 1:]
            mask_t = tf.cast(tf.not_equal(t, 0), tf.float32)
            t = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
            t = t * mask_t[:, :, tf.newaxis]
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return preds

    def val_step(x, t, depth_t):
        preds = model(x, t)
        t = t[:, 1:]
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
            out = preds.numpy().reshape(-1)

            source = ' '.join(en_vocab.decode(source))
            target = ' '.join(ja_vocab.decode(target))
            out = ' '.join(ja_vocab.decode(out))

            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 9:
                break
