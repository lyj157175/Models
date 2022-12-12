import torch
import torch.nn as nn
import numpy as np
import pickle



class Preprocess(nn.Module):
    '''
    代替常见的LSTM，用一种简化的门控机制来提取语义特征
    '''
    def __init__(self, in_feature, out_feature):
        super(Preprocess, self).__init__()
        self.wi = nn.Parameter(torch.randn(in_feature, out_feature))  # embedding_size, hidden_size
        self.bi = nn.Parameter(torch.randn(out_feature))

        self.wu = nn.Parameter(torch.randn(in_feature, out_feature))
        self.bu = nn.Parameter(torch.randn(out_feature))
    
    def froward(self, x):
        print(1111111)
        # x: b, seq_len, embedding
        gate = torch.matmul(x, self.wi)  # b, seq_len, hidden_size
        gate = torch.sigmoid(gate + self.bi.expand_as(gate))   # b, seq_len, hidden_size, bi变为gate形状的tensor

        out = torch.matmul(x, self.wu)   # b, seq_len, hidden_size
        out = torch.tanh(out + self.bu.expand_as(out))  # b, seq_len, hidden_size
        return gate * out  # b, seq_len, hidden_size



class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))

    def forward(self, q, a):
        # q: b, seq_len_1, hidden_size
        # a: b, seq_len_2, hidden_size
        G = torch.matmul(q, self.w)  # b, seq_len_1, hidden_size
        G = G + self.b.expand_as(G) # b, seq_len_1, hidden_size
        G = torch.matmul(G, a.permute(0, 2, 1))  # b, seq_len_1, seq_len_2
        G = torch.softmax(G, dim=1)  # b, seq_len_1, seq_len_2

        H = torch.matmul(G.permate(0, 2, 1), q) # q, seq_len_2, hidden_size
        return H


class Compare(nn.Module):

    def __init__(self, hidden_size):
        super(Compare, self).__init__()
        self.w = nn.Parameter(torch.randn(2*hidden_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))
    
    def forward(self, h, a):
        # h=a:  b, seq_len_2, hidden_size
        sub = (a-h) * (a-h)
        mult = h*a   # b, seq_len_2, hidden_size
        T = torch.matmul(torch.cat([sub, mult], dim=2), self.w)   # b, seq_len_2, 2*hidden_size -> b, seq_len_2, hidden_size
        T = torch.relu(T + self.b.expand_as(T))  # b, seq_len_2, hidden_size
        return T



class Compare_Aggregate(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, seq_len_a, window, class_num):
        super(Compare_Aggregate, self).__init__()
        self.window = window
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.preprocess = Preprocess(embedding_dim, hidden_size)
        self.attention = Attention(hidden_size)
        self.compare = Compare(hidden_size)
        self.aggregate = nn.Conv1d(in_channels=seq_len_a, out_channels=window, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(window*hidden_size, class_num)


    def forward(self, q, a):
        q = self.embed(q)  # b, seq_len_1, embedding_dim
        a = self.embed(a)  # b, seq_len_2, embedding_dim
        # 预处理层
        pre_q = self.preprocess(q)  # b, seq_len_1, hidden
        pre_a = self.preprocess(a)  # b, seq_len_2, hidden
        # 注意力层
        h = self.attention(pre_q, pre_a)   # b, seq_len_2, hidden_size
        # 比较层
        t = self.compare(h, pre_a)   # b, seq_len_2, hidden_size
        # 聚合层
        r = self.aggregate(t)  # b, window, hidden_size

        r = r.view(-1, self.window*self.hidden_size)  
        # 分类层
        out = self.fc(r)  # b, class_num
        return out 


if __name__ == '__main__':
    model = Compare_Aggregate(30000, 100, 100, 20, 3, 3)
    print(model)

