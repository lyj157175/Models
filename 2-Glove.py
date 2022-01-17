import torch
import torch.nn as nn


class Glove(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, x_max, alpha):
        super(Glove, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha

        # 中心词的词向量和中心词的bias
        self.c_embed = nn.Embedding(self.vocab_size, self.embedding_dim).type(torch.float64)
        self.c_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)
        # 周围词的词向量和周围词的bias
        self.p_embed = nn.Embedding(self.vocab_size, self.embedding_dim).type(torch.float64)
        self.p_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)

    def forward(self, c_data, p_data, labels):
        c_data = self.c_embed(c_data)       # b, embedding_dim
        c_data_bias = self.c_bias(c_data)   # b, 1
        p_data = self.p_embed(p_data)       # b, embedding_dim
        p_data_bias = self.p_bias(p_data)   # b, 1
        # 权重的计算利用公式
        weight = torch.pow(labels / self.x_max, self.alpha)
        weight[weight>1] = 1
        # loss的计算使用论文提供的公式
        loss = torch.mean(weight * torch.pow(torch.sum(c_data * p_data, 1) + c_data_bias + p_data_bias - torch.log(labels), 2))
        return loss

    # 保存训练好的glove词向量
    def save_embedding(self, word2idx, file_name):
        embedding1 = self.c_embed.weight.data.cpu().numpy()
        embedding2 = self.p_embed.weight.data.cpu().numpy()
        embedding = (embedding1 + embedding2) / 2
        f = open(file_name, 'w')
        f.write('%d %d\n' % (len(word2idx), self.embedding_dim))
        for w, idx in word2idx.items():
            e = embedding[idx]
            e = ' '.join(map(lambda x: str(x), e))
            f.write('%s %s\n' % (w, e))


if __name__ == '__main__':
    model = Glove(30000, 100, 100, 0.75)
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param)

    word2idx = {'the':0, 'a':1, 'b':2}
    fileaname = 'glove.txt'
    model.save_embedding(word2idx, fileaname)

