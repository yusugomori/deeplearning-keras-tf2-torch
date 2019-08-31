'''
4.1.3 PyTorch（トイ・プロブレム）
'''

import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optimizers


class MLP(nn.Module):
    '''
    多層パーセプトロン
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.a2 = nn.Sigmoid()

        self.layers = [self.l1, self.a1, self.l2, self.a2]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model = MLP(2, 3, 1).to(device)

    '''
    3. モデルの学習
    '''
    criterion = nn.BCELoss()
    optimizer = optimizers.SGD(model.parameters(), lr=0.1)

    def compute_loss(t, y):
        return criterion(y, t)

    def train_step(x, t):
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    epochs = 100
    batch_size = 10
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        train_loss = 0.
        x_, t_ = shuffle(x_train, t_train)
        x_ = torch.Tensor(x_).to(device)
        t_ = torch.Tensor(t_).to(device)

        for n_batch in range(n_batches):
            start = n_batch * batch_size
            end = start + batch_size
            loss = train_step(x_[start:end], t_[start:end])
            train_loss += loss.item()

        print('epoch: {}, loss: {:.3}'.format(
            epoch+1,
            train_loss
        ))

    '''
    4. モデルの評価
    '''
    def test_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.eval()
        preds = model(x)
        loss = compute_loss(t, preds)

        return loss, preds

    loss, preds = test_step(x_test, t_test)
    test_loss = loss.item()
    preds = preds.data.cpu().numpy() > 0.5
    test_acc = accuracy_score(t_test, preds)

    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        test_loss,
        test_acc
    ))
