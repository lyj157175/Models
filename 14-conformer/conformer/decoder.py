# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from torch import Tensor
from typing import Tuple

from conformer.modules import Linear


class DecoderRNNT(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            vocab_size,
            hidden_state_dim,  # 768
            output_dim,  # 512
            num_layers,
            rnn_type='lstm',
            sos_id=1,
            eos_id=2,
            dropout_p=0.2,
    ):
        super(DecoderRNNT, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size, hidden_state_dim)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.out_proj = Linear(hidden_state_dim, output_dim)



    def forward(self, inputs, input_lengths=None, hidden_states=None):
        # inputs: b, seq_len 
        embedded = self.embedding(inputs)  # b, seq_len, hidden_size 

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded.transpose(0, 1), input_lengths.cpu())
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)   # b, seq_len, hidden_size 
            outputs = self.out_proj(outputs.transpose(0, 1))   # b, seq_len, d_model
        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs = self.out_proj(outputs)

        return outputs, hidden_states
