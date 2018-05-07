import torch
from vanilla_lm import Firesuite, to_tensor, to_variable, generation
from review_extractor import indexes_to_characters
from settings import REVIEWS_FOLDER
import numpy as np
import os
import json


if __name__ == '__main__':
    test_transcripts = np.load(os.path.join(REVIEWS_FOLDER, 'train_reviews.npy'))
    dictionary = json.load(open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json')))
    lm = Firesuite(vocab_size=len(dictionary), embedding_size=64, hidden_size=256)
    lm.load_state_dict(torch.load('lm_3.pt'))
    lm.cuda()

    for transcript in test_transcripts:
        inp = transcript[:5]
        print(indexes_to_characters(transcript, dictionary))
        print(inp)
        print(len(generation(inp, 30, lm)[0]))
        print(indexes_to_characters(generation(inp, 30, lm)[0], dictionary))