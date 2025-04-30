import torch
from torch import nn


class BahdanauEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, 
                 decoder_hidden_dim, dropout_p):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.linear = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        '''
        Encode a source sentence. 

        Input:
          - x: a (sequence length, batch size) tensor of token IDs in source language

        Output:
          - outputs: encoder outputs at each time step, given as a tensor of size
            (sequence length, batch size, encoder hidden dim * 2)
          - hidden: final hidden state from RNN, with directions concatenated and
            fed through linear layer; tensor of size (batch size, decoder hidden dim)
        '''

        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)

        hidden = torch.tanh(self.linear(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        ))

        return outputs, hidden