import torch
import torch.nn as nn 


class Seq2Seq(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, hidden_size):
        super(Seq2Seq, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embedding_dim)

        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=4,
                        batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=4,
                        batch_first=True)
        
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, src, tgt, mode='train'):
        src_embed = self.src_embed(src)    # b, max_len, embedding_dim
        # enc_out: b, max_len, hidden_size 
        # enc_hidden: (num_layers*1, b, hidden_size / _)返回每层最后一个时间步的h和c
        enc_out, enc_hidden = self.encoder(src_embed)  

        if mode == 'train':
            tgt_embed = self.tgt_embed(tgt)   # b, max_len, embedding_dim
            # dec_out: b, max_len, hidden_size
            # dec_hidden: (num_layers*1, b, hidden_size / _)
            dec_out, dec_hidden = self.decoder(tgt_embed, enc_hidden)  
            outs = self.fc(dec_out)   # b, max_len, tgt_vovab_size
        else:
            tgt_embed = self.tgt_embed(tgt)  # b, max_len, embedding_dim
            dec_pre_hidden = enc_hidden 
            outs = []
            for i in range(100):
                # dec_out: b, 1, hidden_size
                dec_out, dec_hidden = self.decoder(tgt_embed, dec_pre_hidden)
                out = self.fc(dec_out)  # b, 1, tgt_vocab_size
                pred = torch.argmax(out, dim=-1)  # b, 1
                outs.append(pred.squeeze().cpu().numpy())
                dec_pre_hidden = dec_hidden
                tgt_embed = self.tgt_embed(pred)  
        return outs


if __name__ == '__main__':
    model = Seq2Seq(30000, 30000, 100, 200)
    src = torch.zeros([64, 100]).long()
    tgt = torch.zeros([64, 100]).long()
    out = model(src, tgt, mode='train')
    print(out.shape)
