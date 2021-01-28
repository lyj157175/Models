'''
Word2Vec模型包括：CBOW, Skip-Gram
这里实现的是Skip-Gram + 负采样(NEG)模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size  
        self.embed_size = embed_size  
        # 模型输入，输出是两个一样的矩阵参数nn.Embedding(30000, 100)
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        
        # 初始化
        initrange = 0.5 / self.embed_size
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-initrange, initrange)
        
        
    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围出现过的单词 [batch_size * (c * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (c * 2 * K)]
        return: loss, [batch_size]
        '''
        batch_size = input_labels.size(0) 
        input_embedding = self.in_embed(input_labels) # B * embed_size
        pos_embedding = self.out_embed(pos_labels) # B * (2C) * embed_size 
        neg_embedding = self.out_embed(neg_labels) # B * (2*C*K) * embed_size

        #（b,n,m)*(b,m,p)=(b,n,p)
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze() # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze() # B * (2*C*K)
        
        log_pos = F.logsigmoid(log_pos).sum(1) # batch_size
        log_neg = F.logsigmoid(log_neg).sum(1) # batch_size     
        loss = log_pos + log_neg  # 正样本损失和负样本损失和尽量最大
        return -loss   # batch
    
    # 模型训练有两个矩阵，self.in_embed和self.out_embed, 作者认为输入矩阵比较好
    # 取出输入矩阵参数
    def input_embeddings(self):   
        return self.in_embed.weight.data.cpu().numpy()    



if __name__ == '__main__':
    model = SkipGramModel(30000, 100)
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, ':', param.size())