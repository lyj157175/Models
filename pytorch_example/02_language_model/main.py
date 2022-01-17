# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model




def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)  # 将不足一个batch_szie的数据去掉
    data = data.view(bsz, -1).t().contiguous()  # nbatch, batch_size
    return data.to(device)

def get_batch(seq_len, data, i):
    seq_len = min(seq_len, len(data) - 1 - i)
    src = data[i:i+seq_len]
    target = data[i+1:i+1+seq_len].view(-1)
    if seq_len != 35:
        return None, None
    return src, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)



def train():
    model.train()
    total_loss = 0.
    if model_type != 'transformer':
        hidden = model.init_hidden(batch_size)

    for batch, i in enumerate(range(0, len(train_data - 1), seq_len)):
        # data: seq_len, batch_size
        # target: seq_len * batch_size
        data, target = get_batch(seq_len, train_data, i)
        if data is None:
            continue
        model.zero_grad()
        if model_type == 'transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % 100 == 0:
            train_loss = total_loss / 100
            print('epoch {}  {}/{} batches  lr {:.5f}  train_loss {:5.2f}  ppl {:8.2f}'
                  .format(epoch, batch, len(train_data) // seq_len, lr, train_loss, math.exp(train_loss)))
            total_loss = 0



def evaluate(data_type):
    model.eval()
    total_loss = 0.
    if model_type != 'transformer':
        hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for batch, i in enumerate(range(0, len(data_type - 2), seq_len)):
            data, target = get_batch(seq_len, data_type, i)
            if data is None:
                continue
            if model_type == 'transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output, target)
            total_loss += loss.item()

        val_loss = total_loss / len(val_data)
        print(' val_loss {:5.2f}  ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
        return val_loss







if __name__ == '__main__':
    # 超参设置
    data_path = './data/wikitext-2'
    model_type = 'LSTM'  # RNN_TANH, RNN_RELU, LSTM, GRU, Transformer
    embed_size = 200  # word_embedding
    hidden_size = 200    # hidden units per layer
    nlayers = 2
    lr = 20
    clip = 0.25 # gradient cliping
    epochs = 40
    batch_size = 35
    eval_batch_size = 10
    seq_len = 35  # sequence length
    dropout = 0.2
    tied = 'store_true'  # word embedding and softmax weights
    seed = 123
    log_interval = 200
    onnx_export = ''
    nhead = 2  # the number of heads in the encoder/decoder of the transformer model
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 数据预处理
    corpus = data.Corpus(data_path)
    train_data = batchify(corpus.train, batch_size)  # nbatch, 20
    val_data = batchify(corpus.valid, eval_batch_size)  # nbatch, 10
    test_data = batchify(corpus.test, eval_batch_size)  # nbatch, 10
    ntokens = len(corpus.dictionary)

    if model_type == 'Transformer':
        model = model.TransformerModel(ntokens, embed_size, hidden_size, nhead, nlayers, dropout).to(device)
    else:
        model = model.RNNModel(model_type, embed_size, hidden_size, nlayers, ntokens, dropout).to(device)

    criterion = nn.NLLLoss()

    best_val_loss = float('inf')
    for epoch in range(1, epochs+1):
        train()
        val_loss = evaluate(val_data)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'language_model.pt')
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

        # test
        model = torch.load('language_model.pt')
        test_loss = evaluate(test_data)
        print(' test_loss {:5.2f}  ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))



