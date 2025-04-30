import torch
import torch.nn.functional as F
from torch import nn


class BahdanauAttentionQKV(nn.Module):
    def __init__(self, hidden_size, query_size=None, key_size=None, dropout_p=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_size = hidden_size if query_size is None else query_size

        # assume bidirectional encoder, but can specify otherwise
        self.key_size = 2 * hidden_size if key_size is None else key_size

        self.query_layer = nn.Linear(self.query_size, hidden_size)
        self.key_layer = nn.Linear(self.key_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        '''
        Calculate attention weights using query and key features, with
        an optional mask for input sequences.

        Inputs:
          - hidden: most recent RNN hidden state; (B, Dec)
          - encoder_outputs: RNN outputs at individual time steps with
            directions concatenated; (L, B, 2*Enc)
          - src_mask: boolean tensor of same size as source tokens (Src, B)
            where False denotes tokens to be ignored

        Outputs:
          - attention weights: (B, src) tensor of softmax attention
            weights to be applied to downstream values
        '''

        # (B, H)
        query_out = self.query_layer(hidden)

        # (Src, B, 2*H) --> (Src, B, H)
        key_out = self.key_layer(encoder_outputs)

        # (B, H) + (Src, B, H) = (Src, B, H)
        energy_input = torch.tanh(query_out + key_out)

        # (Src, B, H) --> (Src, B, 1) --> (Src, B)
        energies = self.energy_layer(energy_input).squeeze(2)

        # if a mask is provided, remove masked tokens from softmax calc
        if src_mask is not None:
            energies.data.masked_fill_(src_mask == 0, float("-inf"))

        # softmax over the length dimension
        weights = F.softmax(energies, dim=0)

        # return as (B, Src) as expected by later multiplication
        return weights.transpose(0, 1)