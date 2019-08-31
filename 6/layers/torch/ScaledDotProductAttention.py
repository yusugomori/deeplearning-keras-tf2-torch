'''
6.3.5.1 Scaled Dot-Product Attention - PyTorch
'''

import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 d_k,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.scaler = np.sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        '''
        # Argument
            q, k, v: (batch, sequence, out_features)
            mask:    (batch, sequence)
        '''
        score = torch.einsum('ijk,ilk->ijl', (q, k)) / self.scaler
        score = score - torch.max(score, dim=-1, keepdim=True)[0]

        score = torch.exp(score)
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1).repeat(1, score.size(1), 1)
            score.data.masked_fill_(mask, 0)

        a = score / torch.sum(score, dim=-1, keepdim=True)
        c = torch.einsum('ijk,ikl->ijl', (a, v))

        return c
