from layers.attention import Attention
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding


class MemNet(nn.Module):

    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1 - float(idx + 1) / memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = torch.tensor(weight, dtype=torch.float)
        memory = weight.unsqueeze(2) * memory
        return memory

    def __init__(self, vocab_size, embedding_dim, class_num, hops=6):
        super(MemNet, self).__init__()
        self.hops = hops
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(embedding_dim, score_function='mlp')
        self.x_linear = nn.Linear(embedding_dim, embedding_dim)
        self.dense = nn.Linear(embedding_dim, class_num)

    def forward(self, x, y):
        text_raw_without_aspect_indices, aspect_indices = x, y
        memory_len = torch.sum(text_raw_without_aspect_indices!=0, dim=-1)
        aspect_len = torch.sum(aspect_indices!=0, dim=-1)

        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float)
        memory = self.embed(text_raw_without_aspect_indices)
        memory = self.squeeze_embedding(memory, memory_len)

        # locationed
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        
        for _ in range(self.hops):
            linear_x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + linear_x
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out


