from time import time
import os
from random import sample
from collections import Counter
import json

import numpy as np
from numpy.random import binomial, normal
import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.nn import functional as func
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory

from settings import REVIEWS_FOLDER


class Firesuite(torch.nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, music_features_size=512, unigram_initialization=None):
        super(Firesuite, self).__init__()

        self.music_features_size = music_features_size

        self.activation = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(0.4)

        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=embedding_size, hidden_size=hidden_size),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size * 2),
            nn.LSTM(input_size=hidden_size * 2 + music_features_size, hidden_size=embedding_size)])

        self.locked_dropouts = nn.ModuleList([
            LockedDropout(0.3),
            LockedDropout(0.3),
            LockedDropout(0.4)
        ])

        self.linear = nn.Linear(embedding_size, vocab_size)

        if unigram_initialization is not None:
            self.linear.bias.data = torch.from_numpy(unigram_initialization).float()

        self.rnns_initial_h = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hidden_size)),
            torch.nn.Parameter(torch.zeros(hidden_size * 2)),
            torch.nn.Parameter(torch.zeros(embedding_size)),
        ])

        self.rnns_initial_c = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hidden_size)),
            torch.nn.Parameter(torch.zeros(hidden_size * 2)),
            torch.nn.Parameter(torch.zeros(embedding_size)),
        ])

    def forward(self, features, generate=0, cuda=True):

        N = features.shape[1]
        output = []
        ar = to_variable(torch.zeros(1), cuda=cuda)

        embedded = self.dropout(func.embedding(features, self.linear.weight))
        lstm_out = self.activation(embedded)

        previous_states = [(self.rnns_initial_h[i].expand(N, -1).unsqueeze(0).contiguous(), self.rnns_initial_c[i].expand(N, -1).unsqueeze(0).contiguous()) for i in range(len(self.rnns))]

        for i, rnn in enumerate(self.rnns):
            if i < 2:
                lstm_out, state = rnn(lstm_out, previous_states[i])
                previous_states[i] = state
                lstm_out = self.locked_dropouts[i](lstm_out)
            else:
                zeros = to_variable(torch.zeros(lstm_out.shape[0], lstm_out.shape[1], self.music_features_size))
                lstm_out = torch.cat((lstm_out, zeros), dim=2)
                lstm_out, state = rnn(lstm_out, previous_states[i])
                previous_states[i] = state
                lstm_out = self.locked_dropouts[i](lstm_out)

        ar += torch.sqrt(torch.pow(lstm_out, 2).sum())

        logits = self.linear(lstm_out)

        output.append(logits)
        #        print(torch.max(projected, dim=2)[1].size())

        if (generate > 0):

            new_input = torch.max(logits, dim=2)[1][-1:, :]

            for i in range(generate):

                x = self.dropout(func.embedding(new_input, self.linear.weight))
                lstm_out = self.activation(x)

                for i, rnn in enumerate(self.rnns):
                    lstm_out, state = rnn(lstm_out, previous_states[i])
                    previous_states[i] = state
                    lstm_out = self.locked_dropouts[i](lstm_out)
                ar += torch.sqrt(torch.pow(lstm_out, 2).sum())
                logits = self.linear(lstm_out)

                output.append(logits)
                new_input = torch.max(logits, dim=2)[1]

        logits = torch.cat(output, dim=0)
        return logits, ar

    def set_cuda(self):
        print(torch.cuda.is_available())
        return self.cuda()


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


def lstm_mask(batch_size, embedding_dim, proba):
    mask = binomial(1, proba, batch_size * embedding_dim)
    mask = mask.reshape(1, batch_size, embedding_dim)
    mask = torch.autograd.Variable(torch.from_numpy(mask).float().cuda(), requires_grad=False)
    return mask


def to_tensor(narray):
    return torch.from_numpy(narray)


def to_variable(tensor, cuda=True):
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


class SequenceDataLoader(DataLoader):
    """
    Dataset yields features, assignments, and labels
    """

    def __init__(self, data, batch_size=4, ratio=1):
        super(SequenceDataLoader, self).__init__(data, batch_size=batch_size)

        self.len = (self.dataset.shape[0] - 1) // int(round(self.batch_size * ratio))

    def __iter__(self):
        concatenated_objective = self.dataset[1:self.len * self.batch_size + 1]
        concatenated_data = self.dataset[:self.len * self.batch_size]
        concatenated_objective = concatenated_objective.reshape(self.batch_size, -1).transpose()
        concatenated_data = concatenated_data.reshape(self.batch_size, -1).transpose()
        height = concatenated_objective.shape[0]

        current_stop = 0
        while True:
            if height - current_stop < 35:
                features = to_variable(to_tensor(concatenated_data[current_stop:]).long())
                objective = to_variable(to_tensor(concatenated_objective[current_stop:]).long())
                yield features, objective
                break
            else:
                next_stop = get_next_stop(current_stop)
                features = to_variable(to_tensor(concatenated_data[current_stop:next_stop]).long())
                objective = to_variable(to_tensor(concatenated_objective[current_stop:next_stop]).long())
                yield features, objective
                if next_stop > height:
                    break
                else:
                    current_stop = next_stop

    def __len__(self):
        return self.len


def get_dataset(name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', name)
    data = np.load(path)
    data = np.concatenate(sample(list(data), len(data)))
    return data


def get_next_stop(current_stop):
    jump = 8 + 8 * binomial(1, 0.95, 1)[0]
    jump = normal(jump, 5, 1)[0]
    jump = min(25, jump)
    next_stop = current_stop + jump
    next_stop = int(round(next_stop))
    if next_stop <= current_stop:
        next_stop = current_stop + 15
    return next_stop


def training_routine(num_epochs=7, batch_size=10, reg=0.00001, smoothing=0.2, alpha=0.01):
    train_data = get_dataset(os.path.join(REVIEWS_FOLDER, 'train_reviews.npy'))
    valid_data = get_dataset(os.path.join(REVIEWS_FOLDER, 'dev_reviews.npy'))
    vocab = json.load(open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json')))

    vocab_size = len(vocab)
    word_counts = Counter(train_data)
    word_counts = np.array([word_counts[i] for i in range(vocab_size)], dtype=np.float32)
    word_counts[0] = 0
    total_count = sum(word_counts)
    word_probs = word_counts / float(total_count)
    word_logprobs = np.log((word_probs * (1. - smoothing)) + (smoothing / total_count))

    data_loader = SequenceDataLoader(data=train_data, batch_size=batch_size)
    dev_loader = SequenceDataLoader(data=valid_data, batch_size=batch_size)

    firesuite = Firesuite(embedding_size=64, vocab_size=vocab_size, hidden_size=256, unigram_initialization=word_logprobs)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(firesuite.parameters(), weight_decay=reg)
    firesuite = firesuite.cuda()
    loss_fn = loss_fn.cuda()
    nll = torch.nn.NLLLoss(size_average=True)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    print('starting training')

    for epoch in range(num_epochs):

        full_nll = 0
        start = time()
        losses = []
        valid_losses = []
        firesuite.train()
        total_sequences = 0
        step = 100000
        current_threshold = step
        word_count = 0

        for (data, truth) in data_loader:

            total_sequences += data.shape[0]
            truth = truth.view(-1, )
            output, ar = firesuite(data)
            output = output.view(-1, vocab_size)
            loss = loss_fn(output, truth) + alpha * ar
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optim.step()
            optim.zero_grad()
            if total_sequences > current_threshold:
                print('did {} out of {} in {}'.format(total_sequences, len(data_loader), time() - start))
                current_threshold += step

        firesuite.eval()

        for (data, truth) in dev_loader:
            total_sequences += data.shape[0]
            truth = truth.view(-1, )
            output, ar = firesuite(data)
            output = output.view(-1, vocab_size)
            loss = loss_fn(output, truth)
            probs = logsoftmax(output.view(-1, vocab_size))
            local_nll = nll(probs, truth.view(-1).long())
            full_nll += local_nll.data.cpu().numpy()[0] * data.shape[0] * data.shape[1]
            word_count += data.shape[0] * data.shape[1]
            valid_losses.append(loss.data.cpu().numpy())

        stop = time()
        print("Epoch {} Loss: {:.4f} Accuracy: {} NLL: {} t={}s".format(epoch, np.asscalar(np.mean(losses)),
                                                                        np.asscalar(np.mean(valid_losses)),
                                                                        full_nll / word_count,
                                                                        stop - start))
        print()

        torch.save(firesuite.state_dict(), 'lm_' + str(epoch) + '.pt')

    return firesuite


if __name__ == '__main__':
    trained_firesuite = training_routine(num_epochs=15)
