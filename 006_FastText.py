import torch 
import torch.nn as nn


class FastText(nn.Module):

    def __init__(self, vocab_size, embedding_dim, max_len, num_label):
        super(FastText, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.avg_pool = nn.MaxPool1d(kernel_size=max_len, stride=1)
        self.fc = nn.Linear(embedding_dim, num_label)

    
    def forward(self, x):
        x = torch.Tensor(x).long()  # b, max_len
        x = self.embed(x)      # b, max_len, embedding_dim
        x = x.transpose(2, 1).contiguous()  # b, embedding_dim, max_len
        x = self.avg_pool(x).squeeze()    # b, embedding_dim, 1
        out = self.fc(x)        # b, num_label
        return out 

if __name__ == '__main__':
    model = FastText(3000, 100, 50, 4)
    x = torch.zeros([64, 50])
    out = model(x)
    print(out.shape)

