import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.dropout_wordvector = nn.Dropout(0.4)
        self.lstm0 = nn.LSTM(input_size=embedding_dim,
                             hidden_size=hidden_dim)
        self.dropout_lstm = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(input_size=hidden_dim,
                             hidden_size=hidden_dim)
        self.lstm2 = nn.LSTM(input_size=hidden_dim,
                             hidden_size=embedding_dim)
        self.dropout_out = nn.Dropout(0.4)
        self.linearOut = nn.Linear(in_features=embedding_dim,
                                   out_features=vocab_size)

    def forward(self, x):
        x = F.embedding(x, self.linearOut.weight)
        x = self.dropout_wordvector(x)
        x, _ = self.lstm0(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm1(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)
        x = self.dropout_out(x)
        x = self.linearOut(x)
        return x

    def get_embedding(self):
        return self.linearOut.weight.data
