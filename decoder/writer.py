import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from random import shuffle

from train import generate, transcribe
from generation_dataset import GenerationDataset, generation_collate
from merge_decoder import Decoder
from settings import MTAT_GENERATION_SPLIT


if __name__ == '__main__':
    test_features = np.load(os.path.join(MTAT_GENERATION_SPLIT, 'test_embedded_features.npy'))
    test_descriptions = np.load(os.path.join(MTAT_GENERATION_SPLIT, 'test_descriptions.npy'))
    test_data = GenerationDataset(test_features, test_descriptions)
    vocab = json.load(open(os.path.join(MTAT_GENERATION_SPLIT, 'train_idx_to_word.json')))
    test_vocab = json.load(open(os.path.join(MTAT_GENERATION_SPLIT, 'test_idx_to_word.json')))

    decoder = Decoder(512, 64, len(vocab))
    decoder.load_state_dict(torch.load('decoder_10.pt'))
    decoder.cuda()
    decoder.train()

    indexes = list(range(len(test_data)))
    shuffle(indexes)

    for i in indexes[:20]:
        print(transcribe(test_data[i][1], test_vocab))
        generate(test_data[i][0], vocab, decoder)