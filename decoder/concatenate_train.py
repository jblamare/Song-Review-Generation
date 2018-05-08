from time import time
import os
import json
from collections import Counter
import numpy as np
import random

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from generation_dataset import GenerationDataset, generation_collate
from merge_decoder import Decoder
from settings import batch_size, MTAT_SPLIT_FOLDER, AWS_FOLDER, CLIP_INFO_FILE, MTAT_GENERATION_SPLIT, MTAT_NPY_FOLDER, \
    n_songs, normalization


def to_tensor(narray):
    return torch.from_numpy(narray)


def to_variable(tensor, cuda=True):
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def generate(features_to_test, vocab, decoder):

    output = decoder.generate(to_variable(to_tensor(features_to_test)))
    output = output.cpu().data.numpy()
    print(transcribe(output, vocab))


def transcribe(indexes, vocab):
    return " ".join((vocab[index] for index in indexes))


def training_routine(num_epochs=7, batch_size=32, reg=0.00001, smoothing=0.2):

    train_features = np.load(os.path.join(MTAT_GENERATION_SPLIT, 'train_embedded_features.npy'))
    train_descriptions = np.load(os.path.join(MTAT_GENERATION_SPLIT, 'train_descriptions.npy'))

    valid_features = np.load(os.path.join(MTAT_GENERATION_SPLIT, 'valid_embedded_features.npy'))
    valid_descriptions = np.load(os.path.join(MTAT_GENERATION_SPLIT, 'valid_descriptions.npy'))

    all_text = np.concatenate(train_descriptions)
    word_counts = Counter(all_text)
    vocab_size = len(word_counts)
    word_counts = np.array([word_counts[i] for i in range(vocab_size)], dtype=np.float32)
    total_count = sum(word_counts)
    word_probs = word_counts / float(total_count)
    word_logprobs = np.log((word_probs * (1. - smoothing)) + (smoothing / total_count))

    vocab = json.load(open(os.path.join(MTAT_GENERATION_SPLIT, 'train_idx_to_word.json')))
    valid_vocab =json.load(open(os.path.join(MTAT_GENERATION_SPLIT, 'valid_idx_to_word.json')))

    train_data = GenerationDataset(train_features, train_descriptions)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generation_collate)

    valid_data = GenerationDataset(valid_features, valid_descriptions)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=generation_collate)

    decoder = Decoder(512, 64, vocab_size, word_logprobs)
    decoder = decoder.cuda()

    # Loss and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), weight_decay=reg)

    i = random.randint(0, len(train_data))
    j = random.randint(0, len(valid_data))
    print(i, j)
    print(transcribe(train_descriptions[i][1:], vocab))
    print(transcribe(valid_descriptions[j][1:], valid_vocab))

    for epoch in range(num_epochs):

        start = time()

        train_losses = []
        valid_losses = []

        decoder.train()

        for i, (features, descriptions, lengths) in enumerate(train_loader):

            lengths = [length - 1 for length in lengths]
            features = to_variable(features)
            targets = descriptions[:, 1:]
            descriptions = descriptions[:, :-1]
            descriptions = to_variable(descriptions)
            targets = to_variable(targets)
            targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]

            decoder.zero_grad()
            outputs = decoder(features, descriptions, lengths)
            loss = loss_fn(outputs, targets)
            loss.backward()
            train_losses.append(loss.data.cpu().numpy())
            optimizer.step()

        decoder.eval()

        for i, (features, descriptions, lengths) in enumerate(valid_loader):

            lengths = [length - 1 for length in lengths]
            features = to_variable(features)
            descriptions = descriptions[:, :-1]
            descriptions = to_variable(descriptions)
            targets = pack_padded_sequence(descriptions, lengths, batch_first=True)[0]

            outputs = decoder(features, descriptions, lengths)
            loss = loss_fn(outputs, targets)
            loss.backward()
            valid_losses.append(loss.data.cpu().numpy())

        stop = time()

        print("Epoch {} Train loss: {:.4f} Valid loss: {} t={}s".format(epoch, np.asscalar(np.mean(train_losses)), np.asscalar(np.mean(valid_losses)), stop-start))

        generate(train_data[i][0], vocab, decoder)
        generate(valid_data[j][0], vocab, decoder)

        torch.save(decoder.state_dict(), 'decoder_' + str(epoch) + '.pt')

    return decoder


if __name__ == '__main__':
    decoder = training_routine(num_epochs=20, batch_size=6, reg=0.00001, smoothing=1)
