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

    def generate(self, input, forward=10, m=None, with_gumbel=False, gumbel_weight=1.0):

        if m is None:
            m = Variable(torch.zeros(input.shape[0], input.shape[1],
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
            for i in range(forward):
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

    # def generate(self, text_features, forward, with_gumbel, gumbel_weight=1.0, m=None, cuda=True):
    #     x = torch.from_numpy(np.transpose(text_features)).long()
    #
    #     previous_states = [(None, None), (None, None), (None, None), (None, None)]
    #
    #     for i in range(forward):
    #         x = Variable(x)
    #         if cuda:
    #             x = x.cuda()
    #             if m is not None:
    #                 m = m.cuda()
    #
    #         if m is not None:
    #             expanded_music = m.unsqueeze(1).expand(
    #                 x.shape[0], x.shape[1], -1).contiguous()
    #         else:
    #             expanded_music = Variable(torch.zeros(x.shape[0], x.shape[1],
    #                                      self.music_dim).cuda(), requires_grad=False)
    #
    #         hidden = F.embedding(x, self.linearOut.weight)
    #         previous_states[0] = self.lstm0(hidden)
    #         previous_states[1] = self.lstm1(previous_states[0][0])
    #         previous_states[2] = self.lstm2(previous_states[1][0])
    #         combined = torch.cat((previous_states[2][0], expanded_music), dim=2)
    #         previous_states[3] = self.lstm3(combined)
    #         hidden = self.linearOut(previous_states[3][0])
    #
    #         out = out[-1, :, :]
    #         if with_gumbel:
    #             gumbel = Variable(sample_gumbel(shape=out.size(), out=out.data.new()))
    #             out += gumbel * gumbel_weight
    #         pred = out.data.max(1, keepdim=True)[1]
    #         pred = torch.t(pred)
    #         if i == 0:
    #             generated = pred
    #         else:
    #             generated = torch.cat([generated, pred], dim=0)
    #         x = torch.cat([x.data, pred], dim=0)
    #     return torch.t(generated)


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
