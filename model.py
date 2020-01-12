import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SkipConnBiLSTM(nn.Module):
    def __init__(self, embds_weights, dicts, h, lin_h, **kwargs):
        super(SkipConnBiLSTM, self).__init__()

        # Variable initializations
        h if type(h) == list else list(h)
        lin_h if type(lin_h) == list else list(lin_h)

        self.num_encoding_layers = len(h)
        self.num_lin_layers = len(lin_h)

        self.LHS_max_len = kwargs['LHS_max_len']
        self.RHS_max_len = kwargs['RHS_max_len']
        self.fine_tune = kwargs['fine_tune']
        self.dropout = kwargs['dropout']
        self.ignore_index = kwargs['ignore_index']
        self.activation = kwargs['activation']
        self.shortcuts = kwargs['shortcuts']

        self.word_to_ix, self.label_to_ix = dicts
        self.embd_dim = embds_weights.size(1)

        # Initialize embedding layer
        self.E = nn.Embedding.from_pretrained(embds_weights, freeze=self.fine_tune)

        # Initialize encoder layers
        self.bilstms = []
        for i in range(self.num_encoding_layers):
            if self.shortcuts == 'all':
                self.bilstms.append(nn.LSTM(self.embd_dim + 2 * sum(h[:i]),
                                            h[i],
                                            dropout=self.dropout,
                                            bidirectional=True,
                                            batch_first=True))
            elif self.shortcuts == 'word':
                self.bilstms.append(nn.LSTM(self.embd_dim + (i != 0) * h[i-1],
                                            h[i],
                                            dropout=self.dropout,
                                            bidirectional=True,
                                            batch_first=True))
            else:
                self.bilstms.append(nn.LSTM((i == 0) * self.embd_dim + (i != 0) * h[i-1],
                                            h[i],
                                            dropout=self.dropout,
                                            bidirectional=True,
                                            batch_first=True))

        # Initialize linear layers
        self.linears = []
        for i in range(self.num_lin_layers):
            self.linears.append(nn.Linear((i == 0) * 2 * h[-1] + (i != 0) * lin_h[i-1], lin_h[i]))

    def forward(self, s1, s2, lens1, lens2):
        # Embedding layer
        s1 = self.E(s1)
        s2 = self.E(s2)

        # Encoding layer
        s1 = self.encode(s1, lens1)
        s2 = self.encode(s2, lens2)

        # Max pooling layer
        s1, _ = torch.max(s1, dim=1)
        s2, _ = torch.max(s2, dim=1)

        # Entaliment Layer
        cat_vec = torch.cat((s1, s2), dim=1)
        dis_vec = torch.abs(s1 - s2)
        prod_vec = s1 * s2
        m = torch.cat((cat_vec, dis_vec, prod_vec), dim=1)

        # MLP layer
        output = self.mlp(m)

        return output

    def encode(self, s, lens):

        batch_size = s.size(0)
        absolute_max_seq_len = s.size(1)

        s = s[:, :max(lens)]
        words = s
        for one_layer in self.bilstms:
            packed = nn.utils.rnn.pack_padded_sequence(s, lens.cpu().numpy(), enforce_sorted=False, batch_first=True)
            out, _ = one_layer(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(out)

            if self.shortcuts == 'all':
                s = torch.cat((s, unpacked), dim=2)
            elif self.shortcuts == 'word':
                s = torch.cat((words, unpacked), dim=2)
            # else (self.shortcuts == 'none') then apply no concatenation

        padded = torch.zeros(batch_size, absolute_max_seq_len, s.size(2))
        padded[:, :s.size(1)] = s

        return padded

    def mlp(self, m):
            # Activation after encoding and after each linear other than last one.
        for one_layer in self.linears:
            if self.activation == 'tanh':
                m = F.tanh(m)
            elif self.activation == 'relu':
                m = F.relu(m)
            else:
                raise Exception('invalid activation function. Use "tanh" or "relu"')
            m = one_layer(m)

        return m
