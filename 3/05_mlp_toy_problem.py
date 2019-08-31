'''
3.7.3 簡単な実験
'''

import numpy as np
from models import MLP
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    np.random.seed(123)

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
    # model = MLP(2, 2, 1)
    model = MLP(2, 3, 1)

    '''
    3. モデルの学習
    '''
    def compute_loss(t, y):
        return (-t * np.log(y) - (1 - t) * np.log(1 - y)).sum()

    def train_step(x, t):
        y = model(x)
        for i, layer in enumerate(model.layers[::-1]):
            if i == 0:
                delta = y - t
            else:
                delta = layer.backward(delta, W)

            dW, db = layer.compute_gradients(delta)
            layer.W = layer.W - 0.1 * dW
            layer.b = layer.b - 0.1 * db

            W = layer.W

        loss = compute_loss(t, y)
        return loss

    epochs = 100
    batch_size = 30
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        train_loss = 0.
        x_, t_ = shuffle(x_train, t_train)

        for n_batch in range(n_batches):
            start = n_batch * batch_size
            end = start + batch_size

            train_loss += train_step(x_[start:end],
                                     t_[start:end])

        if epoch % 10 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(
                epoch+1,
                train_loss
            ))

    '''
    4. モデルの評価
    '''
    preds = model(x_test) > 0.5
    acc = accuracy_score(t_test, preds)
    print('acc.: {:.3f}'.format(acc))
