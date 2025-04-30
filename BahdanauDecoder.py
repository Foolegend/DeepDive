import torch
from torch import nn


class BahdanauDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim,
                 decoder_hidden_dim, attention, dropout_p):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.attention = attention # allowing for custom attention
        self.gru = nn.GRU((encoder_hidden_dim * 2) + embedding_dim,
                          decoder_hidden_dim)
        self.out = nn.Linear((encoder_hidden_dim * 2) + embedding_dim + decoder_hidden_dim,
                             output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, encoder_outputs, src_mask=None):
        '''
        Decode an encoder's output.

        B: batch size
        S: source sentence length
        T: target sentence length
        O: output size (target vocab size)
        Enc: encoder hidden dim
        Dec: decoder hidden dim
        Emb: embedding dim

        Inputs:
          - input: a vector of length B giving the most recent decoded token
          - hidden: a (B, Dec) most recent RNN hidden state
          - encoder_outputs: (S, B, 2*Enc) sequence of outputs from encoder RNN

        Outputs:
          - output: logits for next token in the sequence (B, O)
          - hidden: a new (B, Dec) RNN hidden state
          - attentions: (B, S) attention weights for the current token over the source sentence
        '''

        # (B) --> (1, B)
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        attentions = self.attention(hidden, encoder_outputs, src_mask)

        # (B, S) --> (B, 1, S)
        a = attentions.unsqueeze(1)

        # (S, B, 2*Enc) --> (B, S, 2*Enc)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        # weighted encoder representation
        # (B, 1, S) @ (B, S, 2*Enc) = (B, 1, 2*Enc)
        weighted = torch.bmm(a, encoder_outputs)

        # (B, 1, 2*Enc) --> (1, B, 2*Enc)
        weighted = weighted.transpose(0, 1)

        # concat (1, B, Emb) and (1, B, 2*Enc)
        # results in (1, B, Emb + 2*Enc)
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()

        # get rid of empty leading dimensions
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        # concatenate the pieces above
        # (B, Dec), (B, 2*Enc), and (B, Emb)
        # result is (B, Dec + 2*Enc + Emb)
        linear_input = torch.cat((output, weighted, embedded), dim=1)

        # (B, Dec + 2*Enc + Emb) --> (B, O)
        output = self.out(linear_input)

        return output, hidden.squeeze(0), attentions