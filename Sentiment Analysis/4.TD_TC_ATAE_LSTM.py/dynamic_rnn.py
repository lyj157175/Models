import torch
import torch.nn as nn


class DynamicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM': 
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x, x_length):
        '''
        x: b, seq_len
        x_length: b
        '''
        # sort, y=torch.sort(x)  y[0]从小到大排列，y[1]索引
        x_sort_idx = torch.sort(-x_length)[1].long()  # 从大到小
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()   
        x = x[x_sort_idx]
        x_len = x_length[x_sort_idx]

        # pack
        x_pack = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        if self.rnn_type == 'LSTM':
            # out_pack: b, seq_len, hidden_size
            out_pack, (h, c) = self.rnn(x_pack, None)
        else:
            out_pack, h = self.rnn(x_pack, None)
            c = None
        
        # num_layer*num_directional, b, hidden_size -> b, num_layer*num_directional, hidden_size
        h = torch.transpose(h, 0, 1)[x_unsort_idx]   # 

        if self.only_use_last_hidden_state:
            return h
        
        # pad
        else:
            out = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            
        

