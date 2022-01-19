import torch
import torch.nn as nn
from torch import Tensor

from conformer.activation import Swish
from conformer.modules import Linear



class FeedForward(nn.Module):

    def __init__(self, d_model=512,  expansion_factor=4, dropout_p=0.1):
        super(FeedForward, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, d_model * expansion_factor, bias=True),   # b, seq_len, d_model*4
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(d_model * expansion_factor, d_model, bias=True),   # b, seq_len, d_model
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        # inputs: b, seq_len, d_model
        # output: b, seq_len, d_model
        return self.sequential(inputs)