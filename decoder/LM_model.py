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
        if self.training:
            x = self.dropout_lstm(x)
        x, _ = self.lstm1(x)
        if self.training:
            x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)
        combined = torch.cat((x, m), dim=2)
        x, _ = self.lstm3(combined)
        x = self.dropout_out(x)
        x = self.linearOut(x)
        return x

    def get_embedding(self):
        return self.linearOut.weight.data

    def generate(self, input, forward=10, m=None, with_gumbel=False, gumbel_weight=1.0):

        if m is None:
            m = Variable(torch.zeros(input.shape[0], 1,
                                     self.music_dim).cuda(), requires_grad=False)
        else:
            m = m.unsqueeze(1)

        h = input
        h = F.embedding(h, self.linearOut.weight)
        states = []
        h, state = self.lstm0(h)
        states.append(state)
        h, state = self.lstm1(h)
        states.append(state)
        h, state = self.lstm2(h)
        states.append(state)
        combined = torch.cat((h, m.expand(input.shape[0], input.shape[1], -1).contiguous()), dim=2)
        h, state = self.lstm3(combined)
        states.append(state)
        h = self.linearOut(h)
        if with_gumbel:
            gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel * gumbel_weight
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1]
            for i in range(forward - 1):
                h = F.embedding(h, self.linearOut.weight)  # (n, t, c)
                h, states[0] = self.lstm0(h, states[0])
                h, states[1] = self.lstm1(h, states[1])
                h, states[2] = self.lstm2(h, states[2])
                combined = torch.cat((h, m), dim=2)
                h, states[3] = self.lstm3(combined, states[3])
                h = self.linearOut(h)
                if with_gumbel:
                    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel * gumbel_weight
                outputs.append(h)
                h = torch.max(h, dim=2)[1]
            logits = torch.cat([logits] + outputs, dim=1)
        return logits

    # def random_search(self, input, forward=10, m=None, width=10, with_gumbel=False, gumbel_weight=1.0):
    #
    #     best_score = 10E10
    #     best_entry = None
    #     best_X = None
    #     loss_function = nn.CrossEntropyLoss()
    #
    #     for _ in range(width):
    #         logits = self.generate(input, forward=forward, m=m, with_gumbel=with_gumbel, gumbel_weight=gumbel_weight)
    #         entry = torch.max(logits, dim=2)[1].long()[0, :]
    #         X = torch.cat((Variable(torch.from_numpy(np.zeros(1))).long().cuda(), entry[:-1]), dim=0).unsqueeze(1)
    #
    #         if m is None:
    #             expanded_music = Variable(torch.zeros(X.shape[0], X.shape[1],
    #                                      self.music_dim).cuda(), requires_grad=False)
    #         else:
    #             expanded_music = m.unsqueeze(1).expand(X.shape[0], X.shape[1], -1).contiguous()
    #
    #         out = self.forward(X, expanded_music)
    #         out = out.view(out.shape[0] * out.shape[1], out.shape[2])
    #         Y = entry.view(-1)
    #         loss = loss_function(out, Y).data.cpu().numpy()[0]
    #
    #         if loss < best_score:
    #             best_entry = entry
    #             best_X = X.squeeze(1)
    #             best_score = loss
    #
    #     return best_X

    def random_search(self, input, forward=10, m=None, width=10, with_gumbel=False, gumbel_weight=1.0):

        best_score = 10E10
        best_entry = None
        best_X = None
        loss_function = nn.CrossEntropyLoss()

        for _ in range(width):
            logits = self.generate(input, forward=forward, m=m, with_gumbel=with_gumbel, gumbel_weight=gumbel_weight)
            entry = torch.max(logits, dim=2)[1].long()[0, :]
            Y = entry[1:]
            X = entry[:-1].unsqueeze(1)

            if m is None:
                expanded_music = Variable(torch.zeros(X.shape[0], X.shape[1],
                                         self.music_dim).cuda(), requires_grad=False)
            else:
                expanded_music = m.unsqueeze(1).expand(X.shape[0], X.shape[1], -1).contiguous()

            out = self.forward(X, expanded_music)
            out = out.view(out.shape[0] * out.shape[1], out.shape[2])
            loss = loss_function(out, Y).data.cpu().numpy()[0]

            if loss < best_score:
                best_entry = entry
                best_score = loss

        return best_entry


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


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))
