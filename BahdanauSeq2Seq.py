import random

import torch
from torch import nn


class BahdanauSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.tgt_vocab_size = decoder.output_dim

    def forward(self, src, tgt, src_mask=None, teacher_forcing_ratio=0.5, return_attentions=False):

        tgt_length, batch_size = tgt.shape

        # store decoder outputs
        outputs = torch.zeros(tgt_length, batch_size, self.tgt_vocab_size).to(self.device)
        # attentions = torch.zeros(tgt_length, batch_size, )

        encoder_outputs, hidden = self.encoder(src)
        hidden = hidden.squeeze(1) # B, 1, Enc --> B, Enc (if necessary)

        # start with <bos> as the decoder input
        decoder_input = tgt[0, :]
        attentions = []

        for t in range(1, tgt_length):
            decoder_output, hidden, attention = self.decoder(decoder_input, hidden, encoder_outputs, src_mask)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            top_token = decoder_output.max(1)[1]
            decoder_input = (tgt[t] if teacher_force else top_token)
            attentions.append(attention.unsqueeze(-1))

        if return_attentions:
            return outputs, torch.cat(attentions, dim=-1)
        else:
            return outputs