'''
4.1.2 TensorFlow（トイ・プロブレム）
'''

import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


class MLP(Model):
    '''
    多層パーセプトロン
    '''
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_dim, activation='sigmoid')
        self.l2 = Dense(output_dim, activation='sigmoid')

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
    N = 300
    x, t = datasets.make_moons(N, noise=0.3)
    t = t.reshape(N, 1)

    x_train, x_test, t_train, t_test = \
        train_test_split(x, t, test_size=0.2)

    '''
    2. モデルの構築
    '''
    model = MLP(3, 1)

    '''
    3. モデルの学習
    '''
    criterion = losses.BinaryCrossentropy()
    optimizer = optimizers.SGD(learning_rate=0.1)

    def compute_loss(t, y):
        return criterion(t, y)

    def train_step(x, t):
        with tf.GradientTape() as tape:
            preds = model(x)
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    epochs = 100
    batch_size = 10
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        train_loss = 0.
        x_, t_ = shuffle(x_train, t_train)

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            loss = train_step(x_[start:end], t_[start:end])
            train_loss += loss.numpy()

        print('epoch: {}, loss: {:.3}'.format(
            epoch+1,
            train_loss
        ))

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
