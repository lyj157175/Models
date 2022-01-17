import torch 
import torch.nn as nn 
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.max_seq_len = config.max_seq_len

        # 嵌入层
        if config.embedding_pretrain is not None:
            self.word_embed = nn.Embedding.from_pretrained(config.embedding_pretrain, freeze=False)
        else:
            self.word_embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # 卷积层
        # nn.Conv1d(in_channels, out_channels, kernel_size)
        # out: [batch_size, out_channels, n+2p-f/s+1]
        self.conv1d_1 = nn.Conv1d(config.embedding_dim, config.num_filters, config.filters[0])
        self.conv1d_2 = nn.Conv1d(config.embedding_dim, config.num_filters, config.filters[1])
        self.conv1d_3 = nn.Conv1d(config.embedding_dim, config.num_filters, config.filters[2])
        # 池化层
        self.Max_pool_1 = nn.MaxPool1d(self.max_seq_len - config.filters[0] + 1)
        self.Max_pool_2 = nn.MaxPool1d(self.max_seq_len - config.filters[1] + 1)
        self.Max_pool_3 = nn.MaxPool1d(self.max_seq_len - config.filters[2] + 1)
        # dropout层
        self.dropout = nn.Dropout(config.dropout)
        # 分类层
        self.fc = nn.Linear(len(config.filters) * config.num_filters, config.num_label)


    def forward(self, x):
        x = torch.Tensor(x).long()  # b, seq_len
        embed_x = self.word_embed(x)    # b, seq_len, embedding_dim
        embed_x = embed_x.transpose(2, 1).contiguous()  # b, embedding_dim, seq_len
        
        conv_x1 = F.relu(self.conv1d_1(embed_x))  # b, num_filters, n-f+1
        conv_x2 = F.relu(self.conv1d_2(embed_x))  # b, num_filters, n-f+1
        conv_x3 = F.relu(self.conv1d_3(embed_x))  # b, num_filters, n-f+1

        pool_x1 = self.Max_pool_1(conv_x1).squeeze()  # b, num_filters, 1 ---> b, num_filters
        pool_x2 = self.Max_pool_2(conv_x2).squeeze()  # b, num_filters, 1 ---> b, num_filters
        pool_x3 = self.Max_pool_3(conv_x3).squeeze()  # b, num_filters, 1 ---> b, num_filters

        out = torch.cat([pool_x1, pool_x2, pool_x3], dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out 



class Config:
    def __init__(self):
        self.embedding_pretrain = None   # 是否使用预训练词向量
        self.vocab_size = 3000           # 此表大小
        self.embedding_dim = 100         # 词向量维度
        self.filters = [3, 4, 5]         # 卷积核尺寸
        self.num_filters = 100           # 每个尺寸卷积核数量
        self.max_seq_len = 50            # 最大句子长度
        self.dropout = 0.5               # dropout
        self.num_label = 2               # 类别数目
        


if __name__ == '__main__':
    config = Config()
    model = TextCNN(config)
    # print(model)
    x = torch.zeros([64, 50])
    out = model(x)
    print(out.shape)
