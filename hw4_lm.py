import torch
from torch.nn import functional as func
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import random

from ghost_lights import LockedDropout, to_tensor, to_variable, mask, indexes_to_characters
from iron_lstm import Iron


class Volcano(torch.nn.Module):

    def __init__(self, character_number=34, hidden_size=128, embedding_size=256, key_size=128, value_size=128, ratio=1.0,
                 character_initialization=None):
        super(Volcano, self).__init__()

        self.key_size = key_size
        self.value_size = value_size

        self.character_output = torch.nn.Linear(embedding_size, character_number)
        if character_initialization is not None:
            self.character_output.bias.data = torch.from_numpy(character_initialization).float()

        self.initial_context = torch.nn.Parameter(torch.zeros(value_size))

        self.querier_initial_h = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hidden_size)),
            torch.nn.Parameter(torch.zeros(hidden_size)),
            torch.nn.Parameter(torch.zeros(hidden_size)),
        ])

        self.querier_initial_c = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hidden_size)),
            torch.nn.Parameter(torch.zeros(hidden_size)),
            torch.nn.Parameter(torch.zeros(hidden_size)),
        ])

        self.cell0 = torch.nn.LSTMCell(input_size=embedding_size + value_size, hidden_size=hidden_size)
        self.cell1 = torch.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.cell2 = torch.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.query_projection = torch.nn.Linear(hidden_size, key_size)
        self.energy_softmax = torch.nn.Softmax(dim=1)

        self.language_model = [Iron()]
        self.language_model[0].cuda()
        self.language_model[0].load_state_dict(torch.load('lm_12.pt'))
        for param in self.language_model[0].parameters():
            param.requires_grad = False
        self.lm_ratio = ratio

    def attend(self, query, keys, values, audio_mask):

        N = query.shape[0]
        L = keys.shape[2]
        A = self.key_size

        query = query.unsqueeze(1)  # N, A to N, 1, A

        energy = torch.bmm(query, keys)  # N, 1, A * N, A, L to N, L
        energy = energy.squeeze(1) * audio_mask

        attention = self.energy_softmax(energy)
        attention = attention * audio_mask
        normalization = attention.sum(dim=1, keepdim=True)
        if 0 in normalization.data.cpu().numpy():
            raise ValueError("zero in normalization !")
        attention = attention / normalization
        attention = attention.unsqueeze(1)

        current_context = torch.bmm(attention, values)  # N, 1, L * N, L, B to N, B

        return current_context

    def forward(self, transcripts, transcript_lengths, audio_lengths, keys, values):

        N = transcripts.shape[0]
        S = transcripts.shape[1]
        B = self.value_size

        audio_mask = mask(audio_lengths)
        keys = keys.transpose(1, 2)  # N, L, A to N, A, L

        features = func.embedding(transcripts, self.character_output.weight)  # N, S to N, S, embedding_size*

        current_cell_states = [(self.querier_initial_h[i].expand(N, -1), self.querier_initial_c[i].expand(N, -1)) for i
                               in range(len(self.querier_initial_h))]
        current_context = self.initial_context.expand(N, -1)

        logits_sequence = []

        for timestep in range(S):
            current_input = torch.cat((features[:, timestep, :], current_context), dim=1)
            current_cell_states[0] = self.cell0(current_input, current_cell_states[0])
            current_cell_states[1] = self.cell1(current_cell_states[0][0], current_cell_states[1])
            current_cell_states[2] = self.cell2(current_cell_states[1][0], current_cell_states[2])

            query = self.query_projection(current_cell_states[2][0])
            current_context = self.attend(query, keys, values, audio_mask).squeeze(1)

            context_and_query = torch.cat((query, current_context), dim=1)
            logits = self.character_output(context_and_query)
            logits_sequence.append(logits.unsqueeze(1))

        logits_sequence = torch.cat(logits_sequence, dim=1)

        language_model_logits = self.language_model[0](transcripts, transcript_lengths)
        language_model_logit_sequence, _ = pad_packed_sequence(language_model_logits, batch_first=True)

        final_logits = logits_sequence + language_model_logit_sequence * self.lm_ratio
        final_output = pack_padded_sequence(final_logits, transcript_lengths, batch_first=True)

        return final_output

    def greedy_search(self, keys, values, audio_length, max_length=100):

        total_logprob = 0
        outputted_characters = [0]
        current_character_tensor = to_variable(torch.zeros(1, 1).long().cuda())
        maxed_out = True

        audio_mask = mask(audio_length)
        keys = keys.transpose(1, 2)  # N, L, A to N, A, L
        current_cell_states = [(self.querier_initial_h[i].unsqueeze(0), self.querier_initial_c[i].unsqueeze(0)) for i in
                               range(len(self.querier_initial_h))]
        current_context = self.initial_context.unsqueeze(0)

        language_model_states = [(self.language_model[0].rnns_initial_h[i].unsqueeze(0).unsqueeze(0), self.language_model[0].rnns_initial_c[i].unsqueeze(0).unsqueeze(0)) for i in range(4)]

        for i in range(max_length):

            features = func.embedding(current_character_tensor, self.character_output.weight)
            current_input = torch.cat((features[0], current_context), dim=1)
            current_cell_states[0] = self.cell0(current_input, current_cell_states[0])
            current_cell_states[1] = self.cell1(current_cell_states[0][0], current_cell_states[1])
            current_cell_states[2] = self.cell2(current_cell_states[1][0], current_cell_states[2])

            query = self.query_projection(current_cell_states[2][0])
            current_context = self.attend(query, keys, values, audio_mask).squeeze(1)

            context_and_query = torch.cat((query, current_context), dim=1)
            logits = self.character_output(context_and_query).unsqueeze(1)
            lm_logits, language_model_states = self.language_model[0].single_iteration(current_character_tensor, language_model_states)

            final_logits = logits + lm_logits * self.lm_ratio

            _, current_character_tensor = torch.max(final_logits, dim=2)
            current_character_index = current_character_tensor.data.cpu().numpy()[0][0]
            logprob = final_logits.data.cpu().numpy()[0][0][current_character_index]
            total_logprob += logprob
            outputted_characters.append(current_character_index)

            if current_character_index == 1:
                maxed_out = False
                break

        total_logprob /= i
        if maxed_out:
            total_logprob -= 10000

        return outputted_characters, total_logprob

    def random_search(self, keys, values, audio_length, width=10, max_length=100):

        best_sequence, best_logprob = self.greedy_search(keys, values, audio_length)
        softmax = torch.nn.Softmax(dim=2)
        audio_mask = mask(audio_length)
        keys = keys.transpose(1, 2)  # N, L, A to N, A, L

        for _ in range(width):

            total_logprob = 0
            outputted_characters = [0]
            current_character_tensor = to_variable(torch.zeros(1, 1).long().cuda())
            maxed_out = True

            current_cell_states = [(self.querier_initial_h[i].unsqueeze(0), self.querier_initial_c[i].unsqueeze(0)) for i in
                                   range(len(self.querier_initial_h))]
            current_context = self.initial_context.unsqueeze(0)

            language_model_states = [(self.language_model[0].rnns_initial_h[i].unsqueeze(0).unsqueeze(0), self.language_model[0].rnns_initial_c[i].unsqueeze(0).unsqueeze(0)) for i in range(4)]

            for i in range(max_length):

                features = func.embedding(current_character_tensor, self.character_output.weight)
                current_input = torch.cat((features[0], current_context), dim=1)
                current_cell_states[0] = self.cell0(current_input, current_cell_states[0])
                current_cell_states[1] = self.cell1(current_cell_states[0][0], current_cell_states[1])
                current_cell_states[2] = self.cell2(current_cell_states[1][0], current_cell_states[2])

                query = self.query_projection(current_cell_states[2][0])
                current_context = self.attend(query, keys, values, audio_mask).squeeze(1)

                context_and_query = torch.cat((query, current_context), dim=1)
                logits = self.character_output(context_and_query).unsqueeze(1)
                lm_logits, language_model_states = self.language_model[0].single_iteration(current_character_tensor, language_model_states)

                final_logits = logits + lm_logits * self.lm_ratio
                probs = softmax(final_logits).data.cpu().numpy()[0][0]
                current_character_index, prob = sample(probs)
                total_logprob += np.log(prob)

                outputted_characters.append(current_character_index)

                if current_character_index == 1:
                    maxed_out = False
                    break


            total_logprob /= i
            if maxed_out:
                total_logprob -= 10000
            if total_logprob > best_logprob:
                best_sequence = outputted_characters

        return best_sequence

    def set_cuda(self):
        print(torch.cuda.is_available())
        return self.cuda()


def sample(probs):
    runner = 0
    cutoff = random.random()

    for i, prob in enumerate(probs):
        runner += prob
        if runner > cutoff:
            break

    return i, prob
