import torch 
import torch.nn as nn 
from torch.autograd import Variable
import numpy as np 
from torch.nn import functional as F


class HAN_Attention(nn.Module):
    '''层次注意力网络文档分类模型实现，词向量，句子向量'''
    def __init__(self, vocab_size, embedding_dim, gru_size, class_num, weights=None, is_pretrain=False):
        super(HAN_Attention, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        # 词注意力
        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)   # 公式中的u(w)  
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)
        # 句子注意力
        self.sentence_gru = nn.GRU(input_size=2*gru_size, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)
        self.sentence_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)   # 公式中的u(s)
        self.sentence_fc = nn.Linear(2*gru_size, 2*gru_size)
        # 文档分类
        self.class_fc = nn.Linear(2*gru_size, class_num)

    def forward(self, x, use_gpu=False):  # x: b, sentence_num, sentence_len
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)  # b*sentence_num, sentence_len
        embed_x = self.word_embed(x)  # b*sentence_num , sentence_len, embedding_dim
        word_output, word_hidden = self.word_gru(embed_x)  # word_output: b*sentence_num, sentence_len, 2*gru_size
        # 计算u(it)
        word_attention = torch.tanh(self.word_fc(word_output))  # b*sentence_num, sentence_len, 2*gru_size
        # 计算词注意力向量weights: a(it)
        weights = torch.matmul(word_attention, self.word_query)  # b*sentence_num, sentence_len, 1
        weights = F.softmax(weights, dim=1)   # b*sentence_num, sentence_len, 1

        x = x.unsqueeze(2)  # b*sentence_num, sentence_len, 1
        if use_gpu:
            # 去掉x中padding为0位置的attention比重
            weights = torch.where(x!=0, weights, torch.full_like(x, 0, dtype=torch.float).cuda()) #b*sentence_num, sentence_len, 1
        else:
            weights = torch.where(x!=0, weights, torch.full_like(x, 0, dtype=torch.float))
        # 将x中padding后的结果进行归一化处理，为了避免padding处的weights为0无法训练，加上一个极小值1e-4
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)  # b*sentence_num, sentence_len, 1
        
        # 计算句子向量si = sum(a(it) * h(it)) ： b*sentence_num, 2*gru_size -> b*, sentence_num, 2*gru_size
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        sentence_output, sentence_hidden = self.sentence_gru(sentence_vector)  # sentence_output: b, sentence_num, 2*gru_size
        # 计算ui
        sentence_attention = torch.tanh(self.sentence_fc(sentence_output))  # sentence_output: b, sentence_num, 2*gru_size
        # 计算句子注意力向量sentence_weights: a(i)
        sentence_weights = torch.matmul(sentence_attention, self.sentence_query)   # sentence_output: b, sentence_num, 1
        sentence_weights = F.softmax(sentence_weights, dim=1)   # b, sentence_num, 1

        x = x.view(-1, sentence_num, x.size(1))   # b, sentence_num, sentence_len
        x = torch.sum(x, dim=2).unsqueeze(2)  # b, sentence_num, 1
        if use_gpu:
            sentence_weights = torch.where(x!=0, sentence_weights, torch.full_like(x, 0, dtype=torch.float).cuda())  
        else:
            sentence_weights = torch.where(x!=0, sentence_weights, torch.full_like(x, 0, dtype=torch.float))  # b, sentence_num, 1
        sentence_weights = sentence_weights / (torch.sum(sentence_weights, dim=1).unsqueeze(1) + 1e-4)  # b, sentence_num, 1 

        # 计算文档向量v
        document_vector = torch.sum(sentence_weights * sentence_output, dim=1)   # b, sentence_num, 2*gru_size
        document_class = self.class_fc(document_vector)   # b, sentence_num, class_num
        return document_class


if __name__ == '__main__':
    model = HAN_Attention(3000, 200, 50, 4)
    x = torch.zeros(64, 50, 100).long()   # b, sentence_num, sentence_len
    x[0][0][0:10] = 1
    document_class = model(x)
    print(document_class.shape)  # 64, 4





