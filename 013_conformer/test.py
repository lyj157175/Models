import torch
from conformer import Conformer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs = torch.rand(2, 422, 80).to(device)
input_lengths = torch.IntTensor([422, 123]).to(device)
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2], [1, 3, 3, 3, 3, 3, 4, 5, 2, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8]).to(device)

model = Conformer(vocab_size=123,
                input_dim=80,
                encoder_dim=512,
                num_encoder_layers=2,
                decoder_dim=768).to(device)

# Forward
# outputs = model(inputs, input_lengths, targets, target_lengths)
# print(outputs.shape)    # b, seq_len, tgt_len, d_model

# Recognize
# inputs: b, seq_len, feat_dim
# input_lengths: b
outputs = model.recognize(inputs, input_lengths)
print(outputs.shape)


