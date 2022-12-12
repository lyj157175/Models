import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model=512, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)  # max_len, d_model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # max_len, 1
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # 256
        pe[:, 0::2] = torch.sin(position * div)   # max_len, d_model
        pe[:, 1::2] = torch.cos(position * div)   # max_len, d_model
        pe = pe.unsqueeze(0)   # 1, max_len, d_model
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]  # 1, seq_len, d_model


