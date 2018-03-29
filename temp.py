import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from npy_loader import NPY
from model import local_model, global_model
from pub_custom_dataset import CustomDataset
from torchnet.meter import AUCMeter
import matplotlib.pyplot as plt
from settings import number_labels, batch_size, epochs, learning_rate, momentum, MTAT_MP3_FOLDER, MTAT_NPY_FOLDER, MTAT_SPLIT_FOLDER, n_songs, seed, normalization
from preprocessing_librosa import extract_features
import os
import pickle
import sys
import random

# Loading training list
train_list_pub = pickle.load(open(os.path.join(MTAT_SPLIT_FOLDER, 'train_list_pub.cP'), 'rb'))
total_train_size = len(train_list_pub)
index = list(range(total_train_size))

combined = list(zip(train_list_pub, index))
random.seed(seed)
random.shuffle(combined)
train_list_pub[:], index[:] = zip(*combined)


# Loading test list
test_list_pub = pickle.load(open(os.path.join(MTAT_SPLIT_FOLDER, 'test_list_pub.cP'), 'rb'))
total_test_size = len(test_list_pub)


# Loading local models
segment_size_list = [18, 27, 54]
n_inputs = 512+512+768
local_models = []
for segment_size in segment_size_list:
    loc_model = local_model(segment_size).cuda()
    loc_model.load_state_dict(torch.load('local_model_'+str(segment_size)+'.pt'))
    loc_model.eval()
    local_models.append(loc_model)


# Loading global model
model = global_model(n_inputs, 512).cuda()
model.load_state_dict(torch.load('global_model_18_27_54_9051_123.pt'))


# Using the models
for start in range(0, total_test_size, n_songs):
    print("Loading dataset...", start)
    test_features = np.concatenate(
        [np.load(os.path.join(MTAT_NPY_FOLDER, 'testing/'+test_list_pub[i])) for i in range(start, min(start+n_songs, total_test_size))])
    test_labels = np.load(
        os.path.join(MTAT_SPLIT_FOLDER, 'y_test_pub.npy'))[start:min(start+n_songs, total_test_size)]
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
        tags, last_layer_features = model(X)
