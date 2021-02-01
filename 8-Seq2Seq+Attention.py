import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 


class Seq2Seq_attention(nn.Module):
    '''Bahdanau论文中的attention复现'''

    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, lstm_size):
        super(Seq2Seq_attention, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embedding_dim)

        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_size, num_layers=1,
                                bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim+2*lstm_size, hidden_size=lstm_size,
                                num_layers=1, batch_first=True)
        # 注意力机制全连接层
        self.attn_fc_1 = nn.Linear(3*lstm_size, 3*lstm_size)
        self.attn_fc_2 = nn.Linear(3*lstm_size, 1)
        # 分类全连接层
        self.fc_1 = nn.Linear(embedding_dim+2*lstm_size+lstm_size, 2*lstm_size)
        self.fc_2 = nn.Linear(2*lstm_size, tgt_vocab_size)

    
    def attention(self, dec_pre_hidden, enc_output):
        '''
        计算变动的注意力向量ci
        dec_input: b, 1, embedding_dim
        dec_pre_hidden=(h0, c0): [(1, b, lstm_size), (1, b, lstm_size)]
        enc_output（相当于公式中的hj）: b, max_len, 2*lstm_size
        '''

        # 计算si-1
        dec_pre_hidden_h = dec_pre_hidden[0].squeeze(0).unsqueeze(1).repeat(1, 100, 1)  # b, max_len, lstm_dim

        # 计算eij： eij = alpha(si-1, hj) = alpha(si-1, hj)
        attn_input = torch.cat([dec_pre_hidden_h, enc_output], dim=-1)   # b, max_len, 3*lstm_size
        attn_weights = self.attn_fc_2(F.relu(self.attn_fc_1(attn_input)))   # b, max_len, 1

        # 计算aij = sotfmax(eij)
        attn_weights = F.softmax(attn_weights, dim=1)    # b, max_len, 1
        
        # 计算注意力向量ci = sum(aij, hj)
        # (b, max_len, 1) * (b, max_len, 2*lstm_size) == b, max_len, 2*lstm_size
        attn_output = torch.sum(attn_weights * enc_output, dim=1).unsqueeze(1)   # b, 1, 2*lstm_size
        return attn_output

        

    def forward(self, src, tgt, mode='train', is_gpu=True):
        src_embed = self.src_embed(src)     # b, max_len, embedding_dim
        # enc_out: b, max_len, 2*lstm_size
        # enc_hidden = (h, c): (2, b, lstm_size / 2, b, lstm_size)
        enc_output, enc_hidden = self.encoder(src_embed)   

        self.attn_outputs = Variable(torch.zeros(tgt.size(0),
                                                tgt.size(1),
                                                enc_output.size(2)))   # b, max_len, 2*lstm_size
        self.dec_outputs = Variable(torch.zeros(tgt.size(0),
                                                tgt.size(1),
                                                enc_hidden[0].size(2)))  # b, max_len, lstm_size
        if is_gpu:
            self.attn_outputs = self.attn_outputs.cuda()
            self.dec_outputs = self.dec_outputs.cuda()
            

        if mode == 'train':
            tgt_embed = self.tgt_embed(tgt)   # b, max_len, embedding_dim
            # dec_pre_hidden=(h0, c0): [(1, b, lstm_size), (1, b, lstm_size)]
            dec_pre_hidden = [enc_hidden[0][0].unsqueeze(0), enc_hidden[1][0].unsqueeze(0)]
            
            for i in range(100):
                dec_input_embed = tgt_embed[:, i, :].unsqueeze(1)  # b, 1, embedding_dim
                # 计算注意力向量ci
                attn_output = self.attention(dec_pre_hidden, enc_output)   # b, 1, 2*lstm_size

                dec_input_lstm = torch.cat([dec_input_embed, attn_output], dim=2)   # b, 1, embedding_dim+2*lstm_size
                # dec_lstm_output: b, 1, lstm_size
                # dec_pre_hidden=(h, c): ((1, b, lstm_size), (1, b, lstm_size))
                dec_output, dec_hidden = self.decoder(dec_input_lstm, dec_pre_hidden)  

                self.attn_outputs[:, i, :] = attn_output.squeeze()   # b, 2*lstm_size -> b, max_len, 2*lstm_size
                self.dec_outputs[:, i, :] =  dec_output.squeeze()  # b, lstm_size   -> b, max_len, lstm_size
                dec_pre_hidden = dec_hidden
            # class_input: b, max_len, emnbedding_size + 2*lstm_size + lstm_size
            class_input = torch.cat([tgt_embed, self.dec_outputs, self.attn_outputs], dim=2)  
            outs = self.fc_2(F.relu(self.fc_1(class_input)))   # b, max_len, tgt_vocab_size
        else:
            dec_input_embed = self.tgt_embed(tgt)
            dec_pre_hidden = [enc_hidden[0][0].unsqueeze(0), enc_hidden[1][0].unsqueeze(0)]
            outs = []
            for i in range(100):
                attn_output = self.attention(dec_pre_hidden, enc_output)   
                dec_input_lstm = torch.cat([dec_input_embed, attn_output], dim=2)  
                dec_output, dec_hidden = self.decoder(dec_input_lstm, dec_pre_hidden)  
                class_input = torch.cat([dec_input_embed, dec_output, attn_output], dim=2)  
                out = self.fc_2(F.relu(self.fc_1(class_input)))   # b, 1, tgt_vocab_size 
                pred = torch.argmax(out, dim=-1)   # b, 1
                outs.append(pred.squeeze().data.cpu().numpy())  # b
                dec_pre_hidden = dec_hidden
                dec_input_embed = self.tgt_embed(pred)
        return outs  


if __name__ == '__main__':
    model = Seq2Seq_attention(3000, 3000, 200, 100)
    src = torch.zeros(64, 100).long()
    tgt = torch.zeros(64, 100).long()
    out = model(src, tgt, mode='train', is_gpu=False)
    print(out.shape)  # b, max_len, tgt_vocab_size

    tgt_test = torch.zeros(64, 1).long()
    pred = model(src, tgt_test, mode='test', is_gpu=False)
    print(np.array(pred).shape)  # max_len, b

                
            

        
