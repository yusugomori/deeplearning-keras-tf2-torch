'''
5.4.4.2 BiRNN - PyTorch (IMDb)
'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optimizers
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
from callbacks import EarlyStopping


class BiRNN(nn.Module):
    def __init__(self, num_words, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(num_words, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.lstm(h)
        h = self.linear(h[:, -1])
        y = self.sigmoid(h)
        return y.squeeze()  # (batch_size, 1) => (batch_size,)


if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model = BiRNN(num_words, 128).to(device)

    '''
    3. モデルの学習
    '''
    criterion = nn.BCELoss()
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)

    def compute_loss(t, y):
        return criterion(y, t)

    def train_step(x, t):
        x = torch.LongTensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def val_step(x, t):
        x = torch.LongTensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.eval()
        preds = model(x)
        loss = criterion(preds, t)

        return loss, preds

    epochs = 1000
    batch_size = 100
    n_batches_train = x_train.shape[0] // batch_size
    n_batches_val = x_val.shape[0] // batch_size
    es = EarlyStopping(patience=5, verbose=1)

    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        x_, t_ = shuffle(x_train, t_train)

        for batch in range(n_batches_train):
            start = batch * batch_size
            end = start + batch_size
            loss, preds = train_step(x_[start:end], t_[start:end])
            train_loss += loss.item()
            train_acc += \
                accuracy_score(t_[start:end].tolist(),
                               preds.data.cpu().numpy() > 0.5)

        train_loss /= n_batches_train
        train_acc /= n_batches_train

        for batch in range(n_batches_val):
            start = batch * batch_size
            end = start + batch_size
            loss, preds = val_step(x_val[start:end], t_val[start:end])
            val_loss += loss.item()
            val_acc += \
                accuracy_score(t_val[start:end].tolist(),
                               preds.data.cpu().numpy() > 0.5)

        val_loss /= n_batches_val
        val_acc /= n_batches_val

        print('epoch: {}, loss: {:.3}, acc: {:.3f}'
              ', val_loss: {:.3}, val_acc: {:.3f}'.format(
                  epoch+1,
                  train_loss,
                  train_acc,
                  val_loss,
                  val_acc
              ))

        if es(val_loss):
            break

    '''
    4. モデルの評価
    '''
    def test_step(x, t):
        return val_step(x, t)

    loss, preds = test_step(x_test, t_test)
    test_loss = loss.item()
    preds = preds.data.cpu().numpy() > 0.5
    test_acc = accuracy_score(t_test, preds)

    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        test_loss,
        test_acc
    ))
