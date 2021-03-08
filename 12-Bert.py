import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import random
import pickle
import tqdm
import math



# --------------------------------- BertEmbedding ----------------------------------

class BertEmbedding(nn.Module):
    '''
    BertEmbedding包括三部分, 三部分相加并输出:
    1. TokenEmbedding  /  2. PositionalEmbedding  /  3. SegmentEmbedding 
    '''
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout) 
        self.token_embed = TokenEmbedding(vocab_size, embed_size)
        self.position_embed = PositionalEmbedding(embed_size)
        self.segment_embed = SegmentEmbedding(embed_size)

    def forward(self, sequence, segment_label):
        x = self.self.token_embed(sequence) + self.position_embed(sequence) + self.segment_embed(segment_label)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
 

class PositionalEmbedding(nn.Module): 

    def __init__(self, embed_size, max_len=512):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, embed_size)  # max_len, model_dim
        pe.requires_grad = False     
        position = torch.arange(0, max_len).unsqueeze(1)  # max_len, 1
        div = torch.exp(torch.arange(0., embed_size, 2) * (- math.log(10000.) / embed_size))  

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)  # 1, max_len, model_dim
        self.register_buffer('pe', pe)   # 缓存区，不参与计算

    def forward(self, x):  
        return self.pe[:, x.size(1)]  # b, model_dim


class SegmentEmbedding(nn.Embedding):

    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


# --------------------------------- TransformerBlock -------------------------------------------

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden, head, feed_forward_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(hidden, head, dropout=dropout)
        self.feed_forward = FeedForward(hidden, feed_forward_hidden, dropout=dropout)
        self.attn_sublayer = SubLayerConnection(hidden, dropout)
        self.ff_sublayer = SubLayerConnection(hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.attn_sublayer(x, lambda x: self.attention(x, x, x, mask))
        x = self.ff_sublayer(x, self.feed_forward)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_std = s.std(-1, keepdim=True)
        return self.alpha * (x - x_mean) / (x_std + self.eps) + self.beta 


class SubLayerConnection(nn.Module):

    def __init__(self, hidden, dropout):
        super(SubLayerConnection, self).__init__()
        self.layer_norm = LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


def attention(q, k, v, mask=None, dropout=None):
    # q=k=v: b, head, max_len, dk
    # mask: b, max_len, max_len 
    dk = q.size(-1)
    scores = torch.matmul(q * k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # b, head, max_len, max_len

    if mask is not None:
        scores = scores.mask_fill(mask==0, -1e9)  # padding_mask, 极小值填充
    
    attention = F.softmax(scores, dim=-1)  # b, head, max_len, max_len
    if dropout is not None:
        attention = dropout(attention) # b, head, max_len, max_len

    return torch.matmul(attention * v), attention


class MultiHeadedAttention(nn.Module):

    def __init__(self, hidden, head, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.dk = hidden // head
        self.head = head
        self.input_linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output_linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, q, k, v, mask=None):
        # q=k=v: b, max_len, hidden
        batch_size = q.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # b, 1, max_len, max_len  

        # hidden的分头操作 q=k=v: b, head, max_len, dk
        q, k, v = [linear(x).view(batch_size, -1, self.head, self.dk).transpose(1, 2) 
                        for linear, x in zip(self.input_linears, (q, k, v))]
        
        # x: b, head, max_len, dk
        # attn: b, head, max_len, max_len
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.dk)  # b, max_len, hidden
        return self.output_linear(x)  # b, max_len, hidden


class FeedForward(nn.Module):

    def __init__(self, hidden, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = GLUE()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class GLUE(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# ------------------------------------ Bert ----------------------------------

class Bert(nn.Module):
    '''
    BertEmbedding + TransformerBlock    
    '''
    def __init__(self, vocab_size, hidden=768, n_layers=12, head=12, dropout=0.1):
        super(Bert, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.head = head
        self.feed_forward_hidden = hidden*4
        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, head, hidden*4, dropout) for _ in range(n_layers)])
        
    
    def forward(self, x, segment_info):
        # x: b, max_len
        # segment_info: b, max_len 
        mask = (x>0).unsqueeze(1).repeat(1, x.size(1), 1)   # b, max_len, max_len
        x = self.embedding(x, segment_info)   # b, max_leb, embed_size
        for transformer in self.transformer_blocks:
            x = transformer(x, mask) 
        return x

# ---------------------------------- BertDataset ------------------------------------

class BertDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding='utf-8', corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        with open(corpus_path, encoding=encoding) as f:
            self.datas = [line[:-1].split("\t")
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        t1, (t2, is_next_label) = self.datas[item][0], self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = [self.vocab.sos_index] + t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        
        # 如果句子不足进行padding
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)

        output = {
            'bert_input': bert_input,
            'bert_label': bert_label,
            'segment_label': segment_label,
            'is_next': is_next_label
        }
        return {key: torch.tensor(value) for key, value in output.items()}

    
    def random_word(self, sentence):
        tokens = sentence.split()
        sent_label = []   # 0是没有被mask，有数值得是mask得位置

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80%用mask替换
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10%词表中随机一个词替换
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                else:
                # 10%用当前词不变
                    tokens[i] = self.vocab.stoi.get(tokens, self.vocab.unk_index)
                sent_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                sent_label.append(0)
        
        return tokens, sent_label


    def random_sent(self, item):
        if random.random > 0.5:
            return self.datas[item][1], 1    # 是下一句  
        else:
            return self.datas[random.randrange(len(self.datas))][1], 0    # 不是下一句


# --------------------------------------- BertTrainer ---------------------------------

class BertTtrainer:

    def __init__(self, bert, )

if __name__ == '__main__':
    bert = Bert(3000, 768, 12, 12, 0.1)
    # print(bert)





















class Args(object):

    def __init__(self):
        self.train_dataset='data/corpus.small'
        self.test_dataset='data/corpus.small'
        self.vocab_path='data/corpus.small.vocab'
        self.output_path='data/corpus.small.vocab'
        self.hidden=256
        self.layers=8
        self.attn_heads=8
        self.seq_len=20
        self.batch_size=64
        self.epochs=10
        self.num_workers=5
        self.with_cuda=True
        self.log_freq=10
        self.corpus_lines=""
        self.lr=1e-3
        self.adam_weight_decay=0.01
        self.adam_beta1=0.9
        self.adam_beta2=0.999
        
args=Args()




