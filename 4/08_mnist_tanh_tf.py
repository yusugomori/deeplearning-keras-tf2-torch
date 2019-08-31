'''
4.3.1.2 tanh - TensorFlow (MNIST)
'''

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


class DNN(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_dim, activation='tanh')
        self.l2 = Dense(hidden_dim, activation='tanh')
        self.l3 = Dense(hidden_dim, activation='tanh')
        self.l4 = Dense(output_dim, activation='softmax')

        self.ls = [self.l1, self.l2, self.l3, self.l4]

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
    t_train = np.eye(10)[t_train].astype(np.float32)
    t_test = np.eye(10)[t_test].astype(np.float32)

    '''
    2. モデルの構築
    '''
    model = DNN(200, 10)

    '''
    3. モデルの学習
    '''
    criterion = losses.CategoricalCrossentropy()
    optimizer = optimizers.SGD(learning_rate=0.01)
    train_loss = metrics.Mean()
    train_acc = metrics.CategoricalAccuracy()

    def compute_loss(t, y):
        return criterion(t, y)

    def train_step(x, t):
        with tf.GradientTape() as tape:
            preds = model(x)
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(t, preds)

        return loss

    epochs = 30
    batch_size = 100
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        x_, t_ = shuffle(x_train, t_train)

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            train_step(x_[start:end], t_[start:end])

        print('epoch: {}, loss: {:.3}, acc: {:.3f}'.format(
            epoch+1,
            train_loss.result(),
            train_acc.result()
        ))

    '''
    4. モデルの評価
    '''
    test_loss = metrics.Mean()
    test_acc = metrics.CategoricalAccuracy()

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
