import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class Transpose(nn.Module):

    def __init__(self, shape):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class ResidualConnection(nn.Module):

    def __init__(self, module, module_factor=1.0, input_factor=1.0):
        super(ResidualConnection, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs):
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class View(nn.Module):

    def __init__(self, shape, contiguous=False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            x = x.contiguous()
        return x.view(*self.shape)