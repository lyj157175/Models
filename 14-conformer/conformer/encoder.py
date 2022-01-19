import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from conformer.feed_forward import FeedForward
from conformer.attention import MultiHeadedSelfAttention
from conformer.convolution import ConformerConv, Conv2dSubampling
from conformer.modules import ResidualConnection, Linear


class ConformerBlock(nn.Module):

    def __init__(
            self,
            d_model=512,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=31,
            half_step_residual=True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnection(
                module=FeedForward(
                    d_model=d_model,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnection(
                module=MultiHeadedSelfAttention(
                    d_model=d_model,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnection(
                module=ConformerConv(
                    d_model=d_model,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnection(
                module=FeedForward(
                    d_model=d_model,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(d_model),
        )

    def forward(self, inputs):
        return self.sequential(inputs)


class ConformerEncoder(nn.Module):

    def __init__(
            self,
            input_dim=80,
            d_model=512,
            num_layers=12,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            input_dropout_p=0.1,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=31,
            half_step_residual=True,
    ):
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2dSubampling(d_model=1, out_channels=d_model)
        self.input_projection = nn.Sequential(
            Linear(d_model * (((input_dim - 1) // 2 - 1) // 2), d_model),   # d_model * 19 , d_mdoel
            nn.Dropout(p=input_dropout_p),
        )
        self.layers = nn.ModuleList([ConformerBlock(
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])



    def forward(self, inputs, input_lengths):
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)  # b, seq_len/4, d_model*feat_dim/4
        outputs = self.input_projection(outputs)   # b, seq_len, d_model
        for layer in self.layers:
            outputs = layer(outputs)   # b, seq_len, d_model

        return outputs, output_lengths
