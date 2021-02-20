import torch
import torch.nn as nn


class Bi_LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, pad_id, class_num, dropout):
        super(Bi_LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden_size, class_num)


    def forward(self, x, x_length):
        embed_x = self.dropout(self.embed(x))    # b, seq_len, embedding_dim
        pack_x = nn.utils.rnn.pack_padded_sequence(embed_x, x_length, batch_first=True)
        # lstm_out: b, seq_len, hidden_size*2
        # hidden: 2*2, b, hidden_size
        lstm_out, (hidden, cell) = self.lstm(pack_x)  
        out, out_length = nn.utils.rnn.pad_packed_sequence(lstm_out)
        out = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)  # b, 2*hidden_size
        out = self.fc(out)  # b, class_num
        return out

if __name__ == '__main__':
    x = torch.zeros(64, 10).long()
    x_length = torch.full((64,), 10, dtype=torch.float32).long()
    model = Bi_LSTM(3000, 100, 256, 0, 1, 0.5)
    out = model(x, x_length)
    print(out.shape)


