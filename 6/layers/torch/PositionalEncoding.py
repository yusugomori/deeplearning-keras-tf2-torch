'''
6.3.5.3 Positional Encoding - PyTorch
'''

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, output_dim,
                 maxlen=6000,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.maxlen = maxlen
        pe = self.initializer()
        self.register_buffer('pe', pe)

    def forward(self, x, mask=None):
        '''
        # Argument
            x: (batch, sequence)
        '''
        pe = self.pe[:x.size(1), :].unsqueeze(0)
        return x + pe

    def initializer(self):
        pe = \
            np.array([[pos / np.power(10000, 2 * (i // 2) / self.output_dim)
                       for i in range(self.output_dim)]
                      for pos in range(self.maxlen)])

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        return torch.from_numpy(pe).float()
