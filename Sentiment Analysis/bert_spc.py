import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, dropout, bert_dim, class_num):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_dim, class_num)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
