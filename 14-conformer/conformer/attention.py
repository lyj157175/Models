import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from conformer.embedding import PositionalEncoding
from conformer.modules import Linear


class RelativeMultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, num_heads=8, dropout_p=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)   # 64  每个头的维度
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.fc_q = Linear(d_model, d_model)
        self.fc_k = Linear(d_model, d_model)
        self.fc_v = Linear(d_model, d_model)
        self.fc_pos = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))   # 8, 64
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))   # 8, 64
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.fc_out = Linear(d_model, d_model)

    def forward(self, q, k, v, pos_embedding, mask=None):
        batch_size = v.size(0)
        q = self.fc_q(q).view(batch_size, -1, self.num_heads, self.d_head)    # b, seq_len, 8, 64
        k = self.fc_k(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)   # b, 8, seq_len, 64
        v = self.fc_v(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)   # b, 8, seq_len, 64
        pos_embedding = self.fc_pos(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)  # b, seq_len, 8, 64

        # b, 8, seq_len, 64/ * /b, 8, 64, seq_len ==>  b, 8, seq_len, seq_len
        x_score = torch.matmul((q + self.u_bias).transpose(1, 2), k.transpose(2, 3))
        # b, 8, seq_len, 64/ * /b, 8, 64, seq_len ==>  b, 8, seq_len, seq_len
        pos_score = torch.matmul((q + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)   # b, 8, seq_len, seq_len
        score = (x_score + pos_score) / self.sqrt_dim   # b, 8, seq_len, seq_len

        if mask is not None:
            mask = mask.unsqueeze(1)  # b, 1, 1, seq_len
            score.masked_fill_(mask, -1e9)

        score = F.softmax(score, -1)   # b, 8, seq_len, seq_len
        score = self.dropout(score)     # b, 8, seq_len, seq_len

        attn = torch.matmul(score, v).transpose(1, 2)   # b, 8, seq_len, 64 => b, seq_len, 8, 64
        attn = attn.contiguous().view(batch_size, -1, self.d_model)  # b, seq_len, d_model
        return self.fc_out(attn)   # b, seq_len, d_model

    def _relative_shift(self, pos_score):
        # pos_score: b, 8, seq_len, seq_len
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)   # b, 8, seq_len, 1
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)    # b, 8, seq_len, seq_len+1
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)  # b, 8, seq_len+1, seq_len
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)   # b, 8, seq_len, seq_len
        return pos_score


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs, mask=None):
        # inputs: b, seq_len, d_model
        # mask: b, 1, seq_len
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)    # 1, seq_len, d_model
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)  # b, seq_len, d_model

        inputs = self.layer_norm(inputs)
        attn = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)  # b, seq_len, d_model

        return self.dropout(attn)
