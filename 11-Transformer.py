import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math 
import copy
import time
from torch.autograd import Variable
# from IPython.display import Image
# seaborn.set_context(context="talk")

'''
Attention is all you need
Transformer模型复现
数据维度始终保持在：batch_size, max_len, model_dim
'''

# ------------------------------------ 模型组件 -----------------------------------------

# Input Embedding
class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dim):
        super(Embedding, self).__init__()
        self.mdoel_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim)  
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.model_dim)   # b, max_len, model_dim


# Positional Encoding
class PositionalEncoding(nn.Module): 

    def __init__(self, model_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, model_dim)  # max_len, model_dim
        position = torch.arange(0, max_len).unsqueeze(1)  # max_len, 1
        div = torch.exp(torch.arange(0., model_dim, 2) * (- math.log(10000.) / model_dim))  

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)  # 1, max_len, model_dim
        self.register_buffer('pe', pe)   # 缓存区，不参与计算

    def forward(self, x):
        x = x + Variable(self.pe, requires_grad=False)   # b, max_len, model_dim
        return self.dropout(x)


# LayerNorm
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(features))   # 可训练参数
        self.beta = nn.Parameter(torch.zeros(features))   # 可训练参数
        self.eps = eps
    
    def forward(self, x):
        x_mean = x.sum(-1, keepdim=True)
        x_std = x.std(-1, keepdim=True)
        return self.alpha * (x - x_mean) / (x_std + self.eps) + self.beta


# SublaterConnection
class SublayerConnection(nn.Module):

    def __init__(self, model_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))



# Linear + Softmax
class Generator(nn.Module):

    def __init__(self, vocab_size, model_dim):
        super(Generator, self).__init__()
        self.linear = nn.Linear(model_dim, vocab_size)  # b, max_len, vocab_size
    
    def forward(self, x):
        return F.log_sotfmax(self.linear(x), dim=-1)


def clone(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# -------------------------------- self attention & feed forward -------------------------------------------

# Scaled dot product attention: (matmul -> scale -> mask -> softmax -> matmul)
# Attention(Q, K, V) = softmax(Q*K.T / math.sqrt(dk))*V
def attention(q, k, v, mask=None, dropout=None):
    # q=k=v: b, 8, max_len, 64
    dk = q.size(-1)  # 64
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # b, 8, max_len, max_len

    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)  # padding mask, 极小值填充
    
    attention = F.sotfmax(scores, dim=-1)  # 计算出来的注意力权重分数，b, 8, max_len, max_len
    if dropout is not None:
        attention = dropout(attention)   # b, 8, max_len, max_len
    return torch.matmul(attention, v), attention


class MultiHeadedAttention(nn.Module):

    def __init__(self, model_dim, head, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.dk = model_dim // head  # 512//8 = 64
        self.head = head
        self.model_dim = model_dim
        self.linears = clone(nn.Linear(model_dim, model_dim), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn = None
    
    def forward(self, q, k, v, mask=None):
        # q=k=v: b, max_len, model_dim
        if mask is not None:
            mask = mask.unsqueeze(1) # b, 1, max_len, model_dim

        batch_size = q.size(0)

        # 对model_dim进行分头操作
        q, k, v = [linear(x).view(batch_size, -1, self.head, self.dk).transpose(1, 2) 
                for linear, x in zip(self.linears, (q, k, v))]  # q=k=v: b, 8, max_len, 64

        # x: b, 8, max_len, 64
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        # x: b, max_len, model_dim
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.dk) # 多头合并
        return self.linears[-1](x)  # b, max_len, model_dim


class FeedForward(nn.Module):

    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc_1 = nn.Linear(model_dim, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc_2(self.dropout(F.relu(self.fc_1(x))))

# -------------------------------- Encoder & Decoder ----------------------------------------

# Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n)
class Encoder(nn.Module):

    def __init__(self, encoder_layer, n):
        super(Encoder, self).__init__()
        self.encoder_layers = clone(encoder_layer, n)
        self.layer_norm = LayerNorm(encoder_layer.model_dim)
    
    def forward(self, x, mask):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return self.layer_norm(x)


class Encoder_Layer(nn.Module):
    
    def __init__(self, model_dim, self_attn, feed_forward, dropout):
        super(Encoder_Layer, self).__init__()
        self.model_dim = model_dim
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(model_dim, dropout), 2) 
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n)
class Decoder(nn.Module):

    def __init__(self, decoder_layer, n):
        super(Decoder, self).__init__()
        self.decoder_layers = clone(decoder_layer, n)
        self.layer_norm = LayerNorm(decoder_layer.model_dim)

    def forward(self, x, memory, src_mask, tgt_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, src_mask, tgt_mask)
        return self.layer_norm(x)


class Decoder_Layer(nn.Module):

    def __init__(self, model_dim, self_attn, src_attn, feed_forward, dropout):
        super(Decoder_Layer, self).__init__()
        self.model_dim = model_dim
        self.self_attn = self_attn
        self.src_attn = src_attn 
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(model_dim, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):  # memory是encoder的输出
        # self_attetion q=k=v,输入是decoder的embedding, decoder_layer第一层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # soft_attention q!=k=v x是deocder的embedding，m是encoder的输出, decoder_layer第二层
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# Transformer
# class Transformer(nn.Module):
    
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(Transformer, self).__init__()
#         self.encoder = encoder
#         self.decocer = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator
    
#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)

#     def decode(self, memory, tgt, src_mask, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

#     # src=tgt: b, max_len
#     def forward(self, src, src_mask, tgt, tgt_mask):
#         memory = self.encode(src, src_mask)
#         return self.generator(self.decode(memory, src_mask, tgt, tgt_mask))  # b, max_len, vocab_size
    

# def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(h, d_model)
#     ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = Transformer(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
#         nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
#         nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
#         Generator(d_model, tgt_vocab)
#     )
    
#     # 参数初始化
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model

# ------------------------------------------- make a transformer --------------------------------------

class Transformer(nn.Module):
    
    def __init__(self, src_vocab, tgt_vocab, model_dim, head, ff_dim, max_len, dropout, n):
        super(Transformer, self).__init__()
        self.src_embed = nn.Sequential(Embedding(src_vocab, model_dim), 
                                    PositionalEncoding(model_dim, dropout, max_len))
        self.tgt_embed = nn.Sequential(Embedding(tgt_vocab, model_dim), 
                                    PositionalEncoding(model_dim, dropout, max_len))

        self.encoder_layer = Encoder_Layer(model_dim, 
                            MultiHeadedAttention(model_dim, head, dropout), 
                            FeedForward(model_dim, ff_dim, dropout), 
                            dropout)
        self.encoder = Encoder(self.encoder_layer, n)

        self.decoder_layer = Decoder_Layer(model_dim, MultiHeadedAttention(model_dim, head, dropout),
                            MultiHeadedAttention(model_dim, head, dropout), 
                            FeedForward(model_dim, ff_dim, dropout),
                            dropout)
        self.decocer = Decoder(self.decoder_layer, n)

        self.generator = Generator(tgt_vocab, model_dim)
    

    # src=tgt: b, max_len
    def forward(self, src, src_mask, tgt, tgt_mask):
        src_embed = self.src_embed(src, src_mask)
        memory = self.encoder(src_embed)

        tgt_embed = self.tgt_embed(tgt)
        logits = self.decoder(tgt_embed, memory, src_mask, tgt_mask)
        output = self.generator(logits)
        return output     # b, max_len, vocab_size



# ------------------------------------------- mask机制 ----------------------------------

# Transformer涉及padding mask 和 sequence mask
# padding mask 在所有的 scaled dot-product attention 里面都用到
# sequence mask 只有在 decoder 的 self-attention 里面用到

# Padding Mask
# 因为每批次输入序列长度是不一样的就要在较短的序列后面填充0，但这些位置其实是没什么意义的，在做attention计算时不该把注意力放在这些位置上，
# 所以需要在这些位置加上一个非常大的负数(负无穷)，这样的话经过softmax，这些位置的概率就会接近0！
# padding mask每个值都是一个Boolean，值为 false的地方就是我们要进行处理的地方。

# Sequence mask
# sequence mask是为了使decoder不能看见未来的信息，也就是对于一个序列，在t时刻码输出应该只能依赖于 t 时刻之前的输出，
# 因此需要把 t 之后的信息给隐藏起来。具体就是产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上即可。

# decoder的 self-attention，里面使用到的 scaled dot-product attention，同时需要padding mask和sequence mask作为attn_mask，
# 具体实现就是两个mask相加作为attn_mask。其他情况，attn_mask一律等于padding mask。


# 为了避免decoder看到未来信息，影响解码，制作一个下三角矩阵
def squence_mask(size):
    attn_shape = (1, size, size)
    subsquence_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsquence_mask) == 0




if __name__ =='__main__':
    model = Transformer(src_vocab=3000, tgt_vocab=3000, model_dim=512, head=8, 
    ff_dim=2048, max_len=5000, dropout=0.1, n=6)
    print(model)
   

