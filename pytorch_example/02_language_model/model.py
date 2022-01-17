import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class RNNModel(nn.Module):

    def __init__(self, rnn_type, embed_size, hidden_size, nlayers, ntokens, dropout=0.5):
        super(RNNModel, self).__init__()
        self.ntokens = ntokens
        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        self.encoder = nn.Embedding(ntokens, embed_size)
        self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, nlayers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, ntokens)

        if embed_size == hidden_size:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)


    def forward(self, x, hidden):
        embed_x = self.drop(self.encoder(x))  # 8, 20, 128
        out, hidden = self.rnn(embed_x, hidden)   # 8,
        out = self.drop(out)
        out = self.decoder(out)  # b, max_len, ntokens
        out = out.view(-1, self.ntokens)  # b*max_len, ntokens
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.hidden_size),
                    weight.new_zeros(self.nlayers, batch_size, self.hidden_size))
        else:  # GRU
            return weight.new_zeros(self.nlayers, batch_size, self.hidden_size)




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(self.pe[:10])
        x = x + self.pe[:x.size(0), :]  # max_len, b, d_model
        return self.dropout(x)





class TransformerModel(nn.Module):

    def __init__(self, ntokens, embed_size, hidden_size, nhead, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.ntokens = ntokens
        self.embed_size = embed_size
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.embed = nn.Embedding(ntokens, embed_size)
        encoder_layer = TransformerEncoderLayer(embed_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)
        self.decoder = nn.Linear(embed_size, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):  # x: seq_len, b
        device = src.device
        self.mask = self._generate_subsequent_mask(src.size(0)).to(device)

        src = self.embed(src) * math.sqrt(self.embed_size)  # seq_len, b, embed_size
        src = self.pos_encoder(src)   # seq_len, b, embed_size

        enc_out = self.transformer_encoder(src, self.mask)  # seq_len, b, embed_size
        out = self.decoder(enc_out)   # seq_len, b, ntokens
        out = F.log_softmax(out, dim=-1)  # seq_len, b, ntokens
        return out


