import torch
import torch.nn as nn 

class C2W(nn.Module):

    def __init__(self, config):
        super(C2W, self).__init__()
        self.char_hidden_dim = config.char_hidden_dim
        self.word_embed_dim = config.word_embed_dim
        self.max_seq_len = config.max_seq_len
        self.lm_hidden_dim = config.lm_hidden_dim 

        self.char_embed = nn.Embedding(config.n_chars, config.char_embed_dim)

        # 字符双向lstm嵌入层
        self.char_lstm = nn.LSTM(input_size=config.char_embed_dim, hidden_size=config.char_hidden_dim, bidirectional=True, batch_first=True)
        # 语言模型单向lstm
        self.lm_lstm = nn.LSTM(input_size=config.word_embed_dim, hidden_size=config.lm_hidden_dim, batch_first=True)
        
        # 词向量表示
        self.fc1 = nn.Linear(2 * config.char_hidden_dim, config.word_embed_dim) 
        # 映射词表进行分类预测
        self.fc2 = nn.Linear(config.lm_hidden_dim, config.vocab_size)

    def forward(self, x):
        x = torch.Tensor(x).long()   # b, char_seq_len
        embed_x = self.char_embed(x) # b, char_seq_len, char_embed_dim
        char_out, _ = self.char_lstm(embed_x)   # b, char_seq_len, 2*char_hidden_dim
        word_input = torch.cat([char_out[:, -1, 0:self.char_hidden_dim], 
                                char_out[:, 0, self.char_hidden_dim:]], dim=1)   # b, 2*char_hidden_dim
        word_input = self.fc1(word_input)   # b, word_embed_dim
        word_input = word_input.view(-1, self.max_seq_len, self.word_embed_dim)  # b, max_seq_len, word_embed_dim
        word_out, _ = self.lm_lstm(word_input)   # b, max_seq_len, lm_hidden_dim 
        word_out = word_out.contiguous().view(-1, self.lm_hidden_dim)  # b*max_seq_len, lm_hidden_dim  
        out = self.fc2(word_out)  # b*max_seq_len, vocab_size
        return out





class Config:
    def __init__(self):
        self.n_chars = 32         # 字符个数
        self.char_embed_dim = 20  # 字符向量的维度
        self.char_hidden_dim = 30 # 字符的隐藏层神经元数目
        self.word_embed_dim = 50  # 单词的维度
        self.lm_hidden_dim = 100  # 语言模型的隐藏神经元数目
        self.max_seq_len = 10     # 句子的最大单词长度
        self.vocab_size = 1000    # 词表的大小

if __name__ == '__main__':
    config = Config()
    model = C2W(config)
    print(model) 
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)