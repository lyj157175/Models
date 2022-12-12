import torch
import torch.nn as nn
from torch import Tensor

from conformer.decoder import DecoderRNNT
from conformer.encoder import ConformerEncoder
from conformer.modules import Linear


class Conformer(nn.Module):

    def __init__(
            self,
            vocab_size,
            input_dim=80,
            encoder_dim=512,
            decoder_dim=768,
            num_encoder_layers=12,
            num_decoder_layers=1,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            input_dropout_p=0.1,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            decoder_dropout_p=0.1,
            conv_kernel_size=31,
            half_step_residual=True,
            decoder_rnn_type="lstm",
    ):
        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.decoder = DecoderRNNT(
            vocab_size=vocab_size,
            hidden_state_dim=decoder_dim,
            output_dim=encoder_dim,
            num_layers=num_decoder_layers,
            rnn_type=decoder_rnn_type,
            dropout_p=decoder_dropout_p,
        )
        self.fc = Linear(encoder_dim << 1, vocab_size, bias=False)


    def joint(self, encoder_outputs, decoder_outputs):
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)  # b, seq_len, 1, d_model
            decoder_outputs = decoder_outputs.unsqueeze(1)  # b, 1, tgt_len, d_model

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])  # b, seq_len, tgt_len, d_model
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])   # b, seq_len, tgt_len, d_model

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)   # b, seq_len, tgt_len, 2*d_model
        outputs = self.fc(outputs)   # b, seq_len, tgt_len, vocab_size
        return outputs


    def forward(self, inputs, input_lengths, targets, target_lengths):
        encoder_outputs, _ = self.encoder(inputs, input_lengths)    # b, seq_len, d_model
        decoder_outputs, _ = self.decoder(targets, target_lengths)  # b, tgt_len, d_model
        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs  # b, seq_len, tgt_len, vocab_size


    def decode(self, encoder_output, max_length):
        # encoder_output: seq_len, d_model
        pred_tokens, hidden_state = [], None
        decoder_input = encoder_output.new_tensor([[self.decoder.sos_id]], dtype=torch.long)  # [[1]] / 1, 1

        for t in range(max_length):
            # decoder_output: 1, 1, d_model
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_states=hidden_state)
            # d_model / d_model => d_model
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)   # softmax
            pred_token = step_output.argmax(dim=0)     # best_token
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)  # 以最优token继续解码

        return torch.LongTensor(pred_tokens)



    def recognize(self, inputs, input_lengths):
        '''greeady search'''
        # inputs: b, seq_len, feat_dim
        # input_lengths: b
        outputs = []
        # encoder_outputs: b, seq_len, d_model
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)   # seq_len

        for encoder_output in encoder_outputs:
            # 每次解码一个句子
            # encoder_output: seq_len, d_model
            decoded_seq = self.decode(encoder_output, max_length)  # tgt_len, d_model
            outputs.append(decoded_seq)   # b, tgt_len, d_model

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs
