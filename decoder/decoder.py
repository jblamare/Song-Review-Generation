import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as func


class Decoder(torch.nn.Module):

    def __init__(self, features_size, embed_size, hidden_size, vocab_size, unigram_initialization=None):

        super(Decoder, self).__init__()

        self.features_compressor = torch.nn.Linear(features_size, embed_size)
        self.feature_activation = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(0.2)

        self.activation = torch.nn.ReLU()

        self.rnns = torch.nn.ModuleList([
            torch.nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True),
            torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True),
            torch.nn.LSTM(input_size=hidden_size, hidden_size=embed_size, batch_first=True)])

        self.rnn_dropouts = torch.nn.ModuleList([
            LockedDropout(0.5),
            LockedDropout(0.5),
            LockedDropout(0.5)
        ])

        self.linear = torch.nn.Linear(embed_size, vocab_size)

        if unigram_initialization is not None:
            self.linear.bias.data = torch.from_numpy(unigram_initialization).float()

    def forward(self, features, descriptions, lengths):

        features = self.feature_activation(self.dropout(self.features_compressor(features)))

        embeddings = self.activation(func.embedding(descriptions, self.linear.weight))
        final_input = torch.cat((features.unsqueeze(1), embeddings), 1)

        final_input = pack_padded_sequence(final_input, lengths, batch_first=True)

        for i, rnn in enumerate(self.rnns):
            final_input, state = rnn(final_input)
            final_input = pad_packed_sequence(final_input, batch_first=True)[0]
            final_input = self.rnn_dropouts[i](final_input)
            final_input = pack_padded_sequence(final_input, lengths, batch_first=True)

        outputs = self.linear(final_input[0])

        return outputs

    def generate(self, features):

        output_ids = []
        previous_states = [None for _ in self.rnns]
        hidden = features.unsqueeze(0)

        hidden = self.feature_activation(self.features_compressor(hidden))

        for _ in range(30):

            hidden = hidden.unsqueeze(1)

            for i, rnn in enumerate(self.rnns):
                hidden, new_state = rnn(hidden, previous_states[i])
                previous_states[i] = new_state

            outputs = self.linear(hidden.squeeze(1))
            predicted = outputs.max(1)[1]
            output_ids.append(predicted)

            if int(predicted) == 1 or int(predicted) == 2:
                break

            hidden = func.embedding(predicted, self.linear.weight)
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
