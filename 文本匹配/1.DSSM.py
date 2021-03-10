import torch 
import torch.nn as nn


class DSSM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dropout):
        super(DSSM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        a = self.embed(a).sum(1)
        b = self.embed(b).sum(1)

        a = self.dropout(torch.tanh(self.fc1(a)))
        a = self.dropout(torch.tanh(self.fc2(a)))
        a = self.dropout(torch.tanh(self.fc3(a)))

        b = self.dropout(torch.tanh(self.fc1(b)))
        b = self.dropout(torch.tanh(self.fc2(b)))
        b = self.dropout(torch.tanh(self.fc3(b)))

        cosine = torch.cosine_similarity(a, b, dim=1, eps=1e-8)   # 计算两个句子的余弦相似度
        return cosine
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    model = DSSM(30, 100, 0.2)
    model._init_weights()
    print(model)
    