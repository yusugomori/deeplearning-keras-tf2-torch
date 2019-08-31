'''
3.4.3 ロジスティック回帰
'''

import numpy as np


class LogisticRegression(object):
    '''
    ロジスティック回帰
    '''
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim,))
        self.b = 0.

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return sigmoid(np.matmul(x, self.w) + self.b)

    def compute_gradients(self, x, t):
        y = self.forward(x)
        delta = y - t
        dw = np.matmul(x.T, delta)
        db = np.matmul(np.ones(x.shape[0]), delta)

        return dw, db


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    np.random.seed(123)

    '''
    1. データの準備
    '''
    # OR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    t = np.array([0, 1, 1, 1])

    '''
    2. モデルの構築
    '''
    model = LogisticRegression(input_dim=2)

    '''
    3. モデルの学習
    '''
    def compute_loss(t, y):
        return (-t * np.log(y) - (1 - t) * np.log(1 - y)).sum()

    def train_step(x, t):
        dw, db = model.compute_gradients(x, t)
        model.w = model.w - 0.1 * dw
        model.b = model.b - 0.1 * db
        loss = compute_loss(t, model(x))
        return loss

    epochs = 100

    for epoch in range(epochs):
        train_loss = train_step(x, t)  # バッチ学習

        if epoch % 10 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(
                epoch+1,
                train_loss
            ))

    '''
    4. モデルの評価
    '''
    for input in x:
        print('{} => {:.3f}'.format(input, model(input)))
