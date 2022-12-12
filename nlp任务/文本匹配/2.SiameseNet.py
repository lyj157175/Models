import torch
import torch.nn as nn


class SiameseNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, embed_matrix, hidden_size, dropout):
        super(SiameseNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.embed.weight.data.copy_(embed_matrix)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        a = self.embed(a)  # b, max_len, embedding_size
        b = self.embed(b)  # b, max_len, embedding_size

        lstm_a, _ = self.lstm(a)   # b, max_len, hidden_size*2
        lstm_b, _ = self.lstm(b)   # b, max_len, hidden_size*2

        out_a = torch.mean(lstm_a, dim=1)  # b, hidden_size*2
        out_b = torch.mean(lstm_b, dim=1)  # b, hidden_size*2


        out_a = self.dropout(torch.tanh(self.fc(out_a)))
        out_b = self.dropout(torch.tanh(self.fc(out_b)))
        cosine = torch.cosine_similarity(out_a, out_b, dim=1, eps=1e-8)
        return cosine


# 自定义对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__() 
    
    # Ew位余弦相似度
    def forward(self, Ew, y):
        l_1 = 0.25*(1.0-Ew)*(1.0-Ew)
        l_0 = torch.where(Ew<m*torch.ones_like(Ew), torch.full_like(Ew, 0), Ew) \
                            * torch.where(Ew<m*torch.ones_like(Ew), torch.full_like(Ew, 0), Ew)
        
        loss=y*1.0*l_1+(1-y)*1.0*l_0
        return loss.sum()


if __name__ == '__main__':
    embed_matrix = torch.randn(3000, 300)
    model = SiameseNet(3000, 300, embed_matrix, 200, 0.2)
    print(model)




