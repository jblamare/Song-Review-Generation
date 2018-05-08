import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, music_dim):
        super(LanguageModel, self).__init__()

        self.music_dim = music_dim

        self.dropout_wordvector = nn.Dropout(0.4)
        self.dropout_lstm = LockedDropout(0.3)
        self.dropout_out = nn.Dropout(0.4)

        self.lstm0 = nn.LSTM(input_size=embedding_dim,
                             hidden_size=hidden_dim)
        self.lstm1 = nn.LSTM(input_size=hidden_dim,
                             hidden_size=hidden_dim)
        self.lstm2 = nn.LSTM(input_size=hidden_dim,
                             hidden_size=hidden_dim)
        self.lstm3 = nn.LSTM(input_size=hidden_dim + music_dim,
                             hidden_size=embedding_dim)

        self.linearOut = nn.Linear(in_features=embedding_dim,
                                   out_features=vocab_size)

        for p in self.lstm0.parameters():
            p.requires_grad = False
        for p in self.lstm1.parameters():
            p.requires_grad = False
        for p in self.lstm2.parameters():
            p.requires_grad = False
        for p in self.linearOut.parameters():
            p.requires_grad = False

    def forward(self, x, m=None):
        if m is None:
            m = Variable(torch.zeros(x.shape[0], x.shape[1],
                                     self.music_dim).cuda(), requires_grad=False)
        x = F.embedding(x, self.linearOut.weight)
        x = self.dropout_wordvector(x)
        x, _ = self.lstm0(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm1(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)
        combined = torch.cat((x, m), dim=2)
        x, _ = self.lstm3(combined)
        x = self.dropout_out(x)
        x = self.linearOut(x)
        return x

    def get_embedding(self):
        return self.linearOut.weight.data


class LockedDropout(torch.nn.Module):
    def __init__(self, p):
        super(LockedDropout, self).__init__()
        self.proba = p

    def forward(self, input_tensor):
        if not self.training:
            return input_tensor
        m = input_tensor.data.new(1, input_tensor.size(1), input_tensor.size(2)).bernoulli_(1 - self.proba) / (
            1 - self.proba)
        mask = Variable(m, requires_grad=False)
        mask = mask.expand_as(input_tensor)
        return input_tensor * mask
