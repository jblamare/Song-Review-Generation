import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as func


class Decoder(torch.nn.Module):

    def __init__(self, features_size, embed_size, vocab_size, unigram_initialization=None):

        super(Decoder, self).__init__()

        self.dropout = torch.nn.Dropout(0.2)

        self.activation = torch.nn.ReLU()

        self.rnns = torch.nn.ModuleList([
            torch.nn.LSTM(input_size=embed_size, hidden_size=embed_size*2, batch_first=True),
            torch.nn.LSTM(input_size=embed_size*2, hidden_size=embed_size*4, batch_first=True),
            torch.nn.LSTM(input_size=embed_size*4, hidden_size=features_size, batch_first=True)])

        self.rnn_dropouts = torch.nn.ModuleList([
            LockedDropout(0.5),
            LockedDropout(0.5),
            LockedDropout(0.5)
        ])

        self.linear = torch.nn.Linear(embed_size, vocab_size)

        self.merge1 = torch.nn.Linear(features_size * 2, embed_size)
        self.activation1 = torch.nn.ReLU()

        if unigram_initialization is not None:
            self.linear.bias.data = torch.from_numpy(unigram_initialization).float()

    def forward(self, features, descriptions, lengths):

        rnn_input = self.activation(func.embedding(descriptions, self.linear.weight))

        for i, rnn in enumerate(self.rnns):
            rnn_input = pack_padded_sequence(rnn_input, lengths, batch_first=True)
            rnn_input, state = rnn(rnn_input)
            rnn_input = pad_packed_sequence(rnn_input, batch_first=True)[0]
            rnn_input = self.rnn_dropouts[i](rnn_input)

        features = features.unsqueeze(1)
        features = features.expand(features.shape[0], rnn_input.shape[1], features.shape[2])
        merged = torch.cat((rnn_input, features), 2)
        merged = self.linear(self.activation1(self.merge1(merged)))
        merged = pack_padded_sequence(merged, lengths, batch_first=True)
        merged = pad_packed_sequence(merged, batch_first=True)
        merged = pack_padded_sequence(merged[0], lengths, batch_first=True)

        return merged[0]

    def generate(self, features):

        output_ids = []
        previous_states = [None for _ in self.rnns]

        starter = Variable(torch.zeros(1, 1).long().cuda())
        hidden = self.activation(func.embedding(starter, self.linear.weight))
        features = features.unsqueeze(0).unsqueeze(1)

        for _ in range(40):

            for i, rnn in enumerate(self.rnns):
                hidden, new_state = rnn(hidden, previous_states[i])
                previous_states[i] = new_state

            merged = torch.cat((hidden, features), 2)
            merged = self.activation1(self.merge1(merged))
            outputs = self.linear(merged).squeeze(0)

            predicted = outputs.max(1)[1]
            output_ids.append(predicted)

            if int(predicted) == 1 or int(predicted) == 2 and _ > 10:
                break

            hidden = func.embedding(predicted, self.linear.weight)
            hidden = hidden.unsqueeze(0)

        output_ids = torch.cat(output_ids, 0)

        return output_ids


class LockedDropout(torch.nn.Module):
    def __init__(self, p):
        super(LockedDropout, self).__init__()
        self.proba = p

    def forward(self, input_tensor):
        if not self.training:
            return input_tensor
        m = input_tensor.data.new(input_tensor.size(0), 1, input_tensor.size(2)).bernoulli_(1 - self.proba) / (
                    1 - self.proba)
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(input_tensor)
        return input_tensor * mask
