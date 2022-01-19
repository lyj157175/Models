import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from conformer.activation import Swish, GLU
from conformer.modules import Transpose




class ConformerConv(nn.Module):

    def __init__(self, d_model, kernel_size=31, expansion_factor=2, dropout_p=0.1):
        super(ConformerConv, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            Transpose(shape=(1, 2)),    # b, d_model, seq_len
            # b, d_model*factor, seq_len
            nn.Conv1d(d_model, d_model * expansion_factor, kernel_size=1, stride=1, padding=0),
            GLU(dim=1),
            # b, d_model, seq_len
            nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(d_model),
            Swish(),
            # b, d_model, seq_len
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        # inputs: b, seq_len, d_model
        # output: b, seq_len, d_model
        return self.sequential(inputs).transpose(1, 2)


class Conv2dSubampling(nn.Module):

    def __init__(self, d_model, out_channels):
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2),   # b, d_model, seq_len/2, feat_dim/2
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),   # b, d_model, seq_len/4, feat_dim/4
            nn.ReLU(),
        )

    def forward(self, inputs, input_lengths):
        # inputs: b, seq_len, feat_dim
        outputs = self.sequential(inputs.unsqueeze(1)) # b, 1, seq_len, feat_dim ==> b, d_model, seq_len/4, feat_dim/4
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3)   # b, seq_len/4, d_model, feat_dim/4
        # b, seq_len/4, d_model*feat_dim/4
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        output_lengths = (input_lengths >> 2) - 1

        return outputs, output_lengths
