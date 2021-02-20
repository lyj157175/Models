import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, class_num):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, class_num)

    
    def forward(self, x):
        embed_x = self.embed(x)   # b, seq_len, embedding_dim
        # rnn_out: b, seq_len, hidden_size
        # hidden: 1, b, hidden_size 
        rnn_out, hidden = self.rnn(embed_x)
        out = self.fc(hidden.squeeze(0))    # b, class_num
        return out 


if __name__ == '__main__':
    x = torch.zeros(64, 10).long()
    model = RNN(3000, 100, 200, 1)
    # print(model.parameters)
    out = model(x)
    print(out.shape)