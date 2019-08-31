'''
A.3.2 モデルの保存と読み込み - TensorFlow
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


class DNN(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_dim, kernel_initializer='he_normal')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.d1 = Dropout(0.5)
        self.l2 = Dense(output_dim, kernel_initializer='he_normal',
                        activation='softmax')

        self.ls = [self.l1, self.b1, self.a1, self.d1,
                   self.l2]

    def call(self, x):
        for layer in self.ls:
            x = layer(x)

        return x


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
    model = DNN(200, 10)

    '''
    3. モデルの学習・保存
    '''
    criterion = losses.SparseCategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)
    train_loss = metrics.Mean()
    train_acc = metrics.SparseCategoricalAccuracy()
    val_loss = metrics.Mean()
    val_acc = metrics.SparseCategoricalAccuracy()

    def compute_loss(t, y):
        return criterion(t, y)

    @tf.function
    def train_step(x, t):
        with tf.GradientTape() as tape:
            preds = model(x)
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(t, preds)

        return loss

    @tf.function
    def val_step(x, t):
        preds = model(x)
        loss = compute_loss(t, preds)
        val_loss(loss)
        val_acc(t, preds)

    epochs = 10
    batch_size = 100
    n_batches_train = x_train.shape[0] // batch_size
    n_batches_val = x_val.shape[0] // batch_size

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

    model.save_weights('model_tf.h5')  # モデルの重みを保存

    print('model weights saved to: {}'.format('model_tf.h5'))

    '''
    4. モデルの読み込み・評価
    '''
    del model  # これまで学習していたモデルを削除

    model = DNN(200, 10)  # 新しいモデルを初期化
    model.build(input_shape=(None, 784))  # モデルをビルド
    model.load_weights('model_tf.h5')  # 学習済モデルの重みを設定

    print('-' * 20)
    print('model loaded.')

    # テストデータの評価
    test_loss = metrics.Mean()
    test_acc = metrics.SparseCategoricalAccuracy()

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
