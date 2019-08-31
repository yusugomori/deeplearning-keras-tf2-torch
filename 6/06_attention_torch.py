'''
6.2.5.2 Encoder-Decoder (Attention) - PyTorch
'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from layers.torch import Attention
from utils import Vocab
from utils.torch import DataLoader


class EncoderDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 maxlen=20,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, device=device)
        self.decoder = Decoder(hidden_dim, output_dim, device=device)

        self.maxlen = maxlen
        self.output_dim = output_dim

    def forward(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.size(1)
        if target is not None:
            len_target_sequences = target.size(0)
        else:
            len_target_sequences = self.maxlen

        hs, states = self.encoder(source)

        y = torch.ones((1, batch_size),
                       dtype=torch.long,
                       device=self.device)
        output = torch.zeros((len_target_sequences,
                              batch_size,
                              self.output_dim),
                             device=self.device)

        for t in range(len_target_sequences):
            out, states = self.decoder(y, hs, states, source=source)
            output[t] = out

            if use_teacher_forcing and target is not None:
                y = target[t].unsqueeze(0)
            else:
                y = out.max(-1)[1]

        return output


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        len_source_sequences = (x.t() > 0).sum(dim=-1)
        x = self.embedding(x)
        x = pack_padded_sequence(x, len_source_sequences)
        h, states = self.lstm(x)
        h, _ = pad_packed_sequence(h)

        return h, states


class Decoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.attn = Attention(hidden_dim, hidden_dim, device=self.device)
        self.out = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x, hs, states, source=None):
        x = self.embedding(x)
        ht, states = self.lstm(x, states)
        ht = self.attn(ht, hs, source=source)
        y = self.out(ht)

        return y, states


if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_dataloader = DataLoader((x_train, t_train),
                                  batch_first=False,
                                  device=device)
    val_dataloader = DataLoader((x_val, t_val),
                                batch_first=False,
                                device=device)
    test_dataloader = DataLoader((x_test, t_test),
                                 batch_size=1,
                                 batch_first=False,
                                 device=device)

    '''
    2. モデルの構築
    '''
    depth_x = len(en_vocab.i2w)
    depth_t = len(ja_vocab.i2w)

    input_dim = depth_x
    hidden_dim = 128
    output_dim = depth_t

    model = EncoderDecoder(input_dim,
                           hidden_dim,
                           output_dim,
                           device=device).to(device)

    '''
    3. モデルの学習・評価
    '''
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t,
                   teacher_forcing_rate=0.5):
        use_teacher_forcing = (random.random() < teacher_forcing_rate)
        model.train()
        preds = model(x, t, use_teacher_forcing=use_teacher_forcing)
        loss = compute_loss(t.view(-1),
                            preds.view(-1, preds.size(-1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def val_step(x, t):
        model.eval()
        preds = model(x, t, use_teacher_forcing=False)
        loss = compute_loss(t.view(-1),
                            preds.view(-1, preds.size(-1)))

        return loss, preds

    def test_step(x):
        model.eval()
        preds = model(x)
        return preds

    epochs = 30

    for epoch in range(epochs):
        print('-' * 20)
        print('epoch: {}'.format(epoch+1))

        train_loss = 0.
        val_loss = 0.

        for (x, t) in train_dataloader:
            loss, _ = train_step(x, t)
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        for (x, t) in val_dataloader:
            loss, _ = val_step(x, t)
            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print('loss: {:.3f}, val_loss: {:.3}'.format(
            train_loss,
            val_loss
        ))

        for idx, (x, t) in enumerate(test_dataloader):
            preds = test_step(x)

            source = x.view(-1).tolist()
            target = t.view(-1).tolist()
            out = preds.max(dim=-1)[1].view(-1).tolist()

            source = ' '.join(en_vocab.decode(source))
            target = ' '.join(ja_vocab.decode(target))
            out = ' '.join(ja_vocab.decode(out))

            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 9:
                break
