import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class SkipConnBiLSTM(nn.Module):
    def __init__(self, embds_weights, dicts, h, lin_h, **kwargs):
        super(SkipConnBiLSTM, self).__init__()

        self.num_encoding_layers = len(h)
        self.num_lin_layers = len(lin_h)

        self.LHS_max_len = kwargs['LHS_max_len'] if 'LHS_max_len' in kwargs else 401
        self.RHS_max_len = kwargs['RHS_max_len'] if 'RHS_max_len' in kwargs else 70
        self.fine_tune = kwargs['fine_tune'] if 'fine_tune' in kwargs else True
        self.dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.1
        self.ignore_index = kwargs['ignore_index'] if 'ignore_index' in kwargs else 0

        self.word_to_ix, self.label_to_ix = dicts

        self.E = nn.Embedding.from_pretrained(embds_weights, freeze=self.fine_tune)
        self.bilstms = []
        for i in range(self.num_encoding_layers):
            self.bilstms.append(nn.LSTM(len(self.word_to_ix) + 2 * sum(h[:i]),
                                        h[i],
                                        dropout=self.dropout,
                                        bidirectional=True,
                                        batch_first=True))

        self.linears = []
        for i in range(self.num_lin_layers):
            if i == 0:
                self.linears.append(nn.Linear(h[-1], lin_h[i]))
            else:
                self.linears.append(nn.Linear(lin_h[i-1], lin_h[i]))

        self.loss_function = nn.NLLLoss(ignore_index=self.ignore_index)

        # Needs to be in train and lr should be dynamic
        self.lr = kwargs['lr'] if 'lr' in kwargs else 0.0002
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

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
        cat_vec = torch.cat((s1,s2), dim=1)
        dis_vec = torch.abs(s1 - s2)
        prod_vec = s1 * s2
        m = torch.cat((cat_vec, dis_vec, prod_vec), dim=1)

        # MLP layer
        m = self.mlp(m)

    def encode(self, s, lens):
        # Do we need to initialize h_0, c_0 before the seconde encode?
        for one_layer in self.bilstms:
            packed = nn.utils.rnn.pack_padded_sequence(s, lens.cpu().numpy(), enforce_sorted=False, batch_first=True)
            out, _ = one_layer(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(out)
            padded_unpacked = self.repad(unpacked, s.size(0), s.size(1), 2 * s.size(2))
            s = torch.cat((s, padded_unpacked), dim=2)
        return padded_unpacked

    def repad(self, x, *org_sizes):
        padded = torch.zeros(org_sizes[0], org_sizes[1], org_sizes[2])
        padded = padded.transpose(0, 1)
        x = x.transpose(0, 1)
        padded[:x.size(0)] = x
        padded = padded.transpose(0, 1)
        return padded

    # def mlp(self, m):
    #     for
