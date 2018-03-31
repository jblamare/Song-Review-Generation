import os
import pickle
import pandas as pd
import json
from time import time
import numpy as np
import torch
from torch.autograd import Variable
from model import local_model, global_model
import random

from mosestokenizer import MosesTokenizer, MosesSentenceSplitter
from nltk.tokenize import sent_tokenize, word_tokenize

from pub_custom_dataset import CustomDataset
from settings import batch_size, MTAT_SPLIT_FOLDER, TVN_FOLDER, CLIP_INFO_FILE, MTAT_GENERATION_SPLIT, MTAT_NPY_FOLDER, \
    n_songs, normalization


def handle_list(path_list, description_dict, db):
    print(len(path_list))
    last_not_found = None
    last_found = None
    last_description = None
    found_descriptions = []
    found_paths = []
    idx_to_word = set()
    start = time()

    for i, path in enumerate(path_list):
        album_url = db.loc[db.mp3_path == path[:-3] + 'mp3']['url'].values[0]
        if last_not_found == album_url:
            continue
        if last_found == album_url:
            description = last_description
        else:
            try:
                description = description_dict[album_url]
                last_found = album_url
                last_description = description
            except KeyError:
                last_not_found = album_url
                continue
        found_descriptions.append(clean_description(description))
        idx_to_word = idx_to_word.union(found_descriptions[-1])
        found_paths.append(path)

        if i % 1500 == 0:
            print(time() - start)
            print(i)

    idx_to_word.discard('<START_TOKEN>')
    idx_to_word.discard('<STOP_TOKEN>')
    idx_to_word.discard('.')
    idx_to_word = ['<START_TOKEN>', '<STOP_TOKEN>', '.'] + sorted(idx_to_word)
    print(idx_to_word)
    word_to_idx = {word: index for index, word in enumerate(idx_to_word)}
    found_descriptions = np.asarray(
        [[word_to_idx[token] for token in description] for description in found_descriptions])

    return found_descriptions, found_paths, idx_to_word, word_to_idx


def handle_set(name, output=True):
    list_pub = pickle.load(open(os.path.join(MTAT_SPLIT_FOLDER, name + '_list_pub.cP'), 'rb'))
    found_descriptions, found_paths, idx_to_word, word_to_idx = handle_list(list_pub, description_dict, db)
    start = time()
    features_npy = compute_npy_path_list(found_paths, name)
    print(time() - start)
    if output:
        print('ouptutting ' + name + ' data')
        np.save(os.path.join(MTAT_GENERATION_SPLIT, name + '_embedded_features.npy'), features_npy)
        np.save(os.path.join(MTAT_GENERATION_SPLIT, name + '_descriptions.npy'), found_descriptions)
        pickle.dump(found_paths, open(os.path.join(MTAT_GENERATION_SPLIT, name + '_found_paths.cP'), 'wb'))
        json.dump(idx_to_word, open(os.path.join(MTAT_GENERATION_SPLIT, name + '_idx_to_word.json'), 'w'),
                  ensure_ascii=False, indent=2)
        json.dump(word_to_idx, open(os.path.join(MTAT_GENERATION_SPLIT, name + '_word_to_idx.json'), 'w'),
                  ensure_ascii=False, indent=2)


def compute_npy_path_list(path_list, name):

    total_test_size = len(path_list)
    outputs = []

    for start in range(0, total_test_size, n_songs):

        print("Loading dataset...", start)
        test_features = np.concatenate(
            [np.load(os.path.join(MTAT_NPY_FOLDER, folder_name(name) + path_list[i])) for i in
             range(start, min(start + n_songs, total_test_size))])
        test_labels = np.load(
            os.path.join(MTAT_SPLIT_FOLDER, 'y_train_pub.npy'))[start:min(start + n_songs, total_test_size)]

        if normalization:
            mean = np.mean(test_features, axis=0)
            var = np.var(test_features, axis=0)
            test_features = (test_features - mean) / np.sqrt(var)

        test_data = CustomDataset(test_features, test_labels)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        print("Dataset loaded")

        for data, labels in test_loader:
            X = Variable(data[:, :, :1242]).cuda()
            X = torch.cat([loc_model(X)[1] for loc_model in local_models], dim=1)
            _, last_layer_features = model(X)
            outputs.append(last_layer_features.cpu().data.numpy())

    return np.concatenate(outputs)


def folder_name(name):
    if name == 'train':
        return 'training/'
    if name == 'test':
        return 'testing/'
    if name == 'valid':
        return 'validation/'
    raise ValueError


def clean_description(text):
    return ['<START_TOKEN>'] + [word.lower() for word in word_tokenize(text)] + ['<STOP_TOKEN>']


if __name__ == '__main__':

    segment_size_list = [18, 27, 54]
    n_inputs = 512+512+768
    local_models = []
    for segment_size in segment_size_list:
        loc_model = local_model(segment_size).cuda()
        loc_model.load_state_dict(torch.load('local_model_'+str(segment_size)+'.pt'))
        loc_model.eval()
        local_models.append(loc_model)
    model = global_model(n_inputs, 512).cuda()
    model.load_state_dict(torch.load('global_model_18_27_54_9051_123.pt'))

    db = pd.read_csv(CLIP_INFO_FILE, sep="\t")
    description_dict = json.load(open(os.path.join(TVN_FOLDER, 'descriptions.json')))

    for name in ['train', 'test', 'valid']:
        handle_set(name, output=True)

    # test_list_pub = pickle.load(open(os.path.join(MTAT_SPLIT_FOLDER, 'test_list_pub.cP'), 'rb'))
    # found_descriptions, found_paths, idx_to_word, word_to_idx = handle_list(test_list_pub, description_dict, db)
    #
    # valid_list_pub = pickle.load(open(os.path.join(MTAT_SPLIT_FOLDER, 'valid_list_pub.cP'), 'rb'))
    # found_descriptions, found_paths, idx_to_word, word_to_idx = handle_list(valid_list_pub, description_dict, db)
