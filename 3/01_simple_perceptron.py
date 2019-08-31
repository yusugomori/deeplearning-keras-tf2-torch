'''
3.3.2 単純パーセプトロン
'''

import numpy as np


class SimplePerceptron(object):
    '''
    単純パーセプトロン
    '''
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim,))
        self.b = 0.

    def forward(self, x):
        y = step(np.matmul(self.w, x) + self.b)
        return y

    def compute_deltas(self, x, t):
        y = self.forward(x)
        delta = y - t
        dw = delta * x
        db = delta

        return dw, db


def step(x):
    return 1 * (x > 0)


if __name__ == '__main__':
    np.random.seed(123)  # 乱数シード

    '''
    1. データの準備
    '''
    d = 2   # 入力次元
    N = 20  # 全データ数

    mean = 5

    x1 = np.random.randn(N//2, d) + np.array([0, 0])
    x2 = np.random.randn(N//2, d) + np.array([mean, mean])

    t1 = np.zeros(N//2)
    t2 = np.ones(N//2)

    x = np.concatenate((x1, x2), axis=0)  # 入力データ
    t = np.concatenate((t1, t2))          # 出力データ

    '''
    2. モデルの構築
    '''
    model = SimplePerceptron(input_dim=d)

    '''
    3. モデルの学習
    '''
    def compute_loss(dw, db):
        return all(dw == 0) * (db == 0)

    def train_step(x, t):
        dw, db = model.compute_deltas(x, t)
        loss = compute_loss(dw, db)
        model.w = model.w - dw
        model.b = model.b - db

        return loss

    while True:
        classified = True
        for i in range(N):
            loss = train_step(x[i], t[i])
            classified *= loss
        if classified:
            break

    '''
    4. モデルの評価
    '''
    print('w:', model.w)  # => w: [1.660725   1.49465147]
    print('b:', model.b)  # => b: -10.0

    print('(0, 0) =>', model.forward([0, 0]))  # => 0
    print('(5, 5) =>', model.forward([5, 5]))  # => 1
