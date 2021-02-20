import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


class BiLSTM_CRF(nn.Module):
    
    def __init__(self, vocab_size, tag2idx, embedding_dim, hidden_size, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.tag2idx = tag2idx
        self.tag_size = len(tag2idx)
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        # 将BiLSTM提取的特征向量映射到特征空间，即经过全连接得到发射分数
        self.fc = nn.Linear(2*hidden_size, self.tag_size)

        # 转移矩阵的参数初始化，transitions[i,j]代表的是从第j个tag转移到第i个tag的转移分数
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        # 其他tag转移到satrt不可能, end转移到其他tag不可能
        self.transitions.data[tag2idx['start'], :] = -10000
        self.transitions.data[:, tag2idx['end']] = -10000

    def _hidden_init(self):
        # 初始化LSTM的参数
        return (torch.randn(2, self.batch_size, self.hidden_size), 
                torch.randn(2, self.batch_size, self.hidden_size))


    def _score_sentence(self, feats, tags):
        # 计算给定tag序列的分数，即一条路径的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # 递推计算路径分数：转移分数 + 发射分数
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _forward_alg(self, feats):
        # 通过前向算法递推计算
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 初始化step 0即START位置的发射分数，START_TAG取0其他位置取-10000
        init_alphas[0][self.tag2idx['start']] = 0.
        # 将初始化START位置为0的发射分数赋值给previous
        previous = init_alphas

        # 迭代整个句子
        for obs in feats:
            # 当前时间步的前向tensor
            alphas_t = []
            for next_tag in range(self.tag_size):
                # 取出当前tag的发射分数，与之前时间步的tag无关
                emit_score = obs[next_tag].view(1, -1).expand(1, self.tag_size)
                # 取出当前tag由之前tag转移过来的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # 当前路径的分数：之前时间步分数 + 转移分数 + 发射分数
                next_tag_var = previous + trans_score + emit_score
                # 对当前分数取log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新previous 递推计算下一个时间步
            previous = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移到STOP_TAG
        terminal_var = previous + self.transitions[self.tag2idx['end']]
        # 计算最终的分数
        scores = log_sum_exp(terminal_var)
        return scores


    def neg_log_likelihood(self, sentence, tags):
        # CRF损失函数由两部分组成，真实路径的分数和所有路径的总分数。
        # 真实路径的分数应该是所有路径中分数最高的。
        # log真实路径的分数/log所有可能路径的分数，越大越好，构造crf loss函数取反，loss越小越好
        self.hidden = self._hidden_init()
        embed_x = self.word_embed(sentence)    # b, max_len, emebdding_dim
        lstm_out, self.hidden = self.lstm(embed_x, self.hidden)  # lstm_out: b, max_len, 2*hidden_size
        lstm_out = lstm_out.view(lstm_out.size(1), lstm_out.size(2))
        feats = self.fc(lstm_out)  # b, max_len, tag_size

        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def _viterbi_decode(self, feats):  # feats: b, max_len, tag_size 
        backpointers = []
        
        # 初始化viterbi的previous变量
        init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.tag2idx['start']] = 0

        previous = init_vvars
        for obs in feats: 
            # 保存当前时间步的回溯指针
            bptrs_t = []
            # 保存当前时间步的viterbi变量
            viterbivars_t = []  

            for next_tag in range(self.tag_size):
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
                # 并不取决与当前tag的发射分数
                next_tag_var = previous + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                
            # 更新previous，加上当前tag的发射分数obs
            previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
            # 回溯指针记录当前时间步各个tag来源前一步的tag
            backpointers.append(bptrs_t)

        # 考虑转移到STOP_TAG的转移分数
        terminal_var = previous + self.transitions[self.tag2ix['end']]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
     
        # 去除START_TAG
        start = best_path.pop()
        assert start == self.tag2idx['start']  
        best_path.reverse()
        return path_score, best_path


    def forward(self, x):  # b, max_len
        # 通过BiLSTM提取发射分数
        self.hidden = self._hidden_init()
        embed_x = self.word_embed(x)    # b, max_len, emebdding_dim
        lstm_out, self.hidden = self.lstm(embed_x, self.hidden)  # lstm_out: b, max_len, 2*hidden_size
        lstm_out = lstm_out.view(lstm_out.size(1), lstm_out.size(2))
        lstm_feats = self.fc(lstm_out)  # b, max_len, tag_size

        # 根据发射分数以及转移分数，通过viterbi解码找到一条最优路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

# --------------------------- utils ------------------------------------
def argmax(vec):
    # 返回vec的dim为1维度上的最大值索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    # 将句子转化为ID
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 前向算法是不断累积之前的结果，这样就会有个缺点
# 指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
# 为了避免这种情况，用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
# SUM = log(exp(s1)+exp(s2)+...+exp(s100))
#     = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
#     = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
# where clip=max
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))




if __name__ == '__main__':
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4
    BATCH_SIZE = 1
    # 构造训练数据
    training_data = [("the wall street journal reported today that apple corporation made money".split(),
                    "B I I I O O O B I O O".split()), 
                    ("georgia tech is a university in georgia".split(),
                    "B I O O O O B".split())]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag2idx = {"B": 0, "I": 1, "O": 2, 'start': 3, 'end': 4}

    model = BiLSTM_CRF(len(word_to_ix), tag2idx, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


    # 训练前检查模型预测结果
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag2idx[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    for epoch in range(300):  
        for sentence, tags in training_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag2idx[t] for t in tags], dtype=torch.long)

            loss = model.neg_log_likelihood(sentence_in, targets)

            loss.backward()
            optimizer.step()

    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(precheck_sent))





