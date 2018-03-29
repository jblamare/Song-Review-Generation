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


def train(segment_size_list):
    train_list_pub = pickle.load(open(os.path.join(MTAT_SPLIT_FOLDER, 'train_list_pub.cP'), 'rb'))
    total_train_size = len(train_list_pub)
    index = list(range(total_train_size))

    combined = list(zip(train_list_pub, index))
    random.seed(seed)
    random.shuffle(combined)
    train_list_pub[:], index[:] = zip(*combined)

    n_inputs = 0
    for segment_size in segment_size_list:
        if segment_size == 18:
            n_inputs += 512
        if segment_size == 27:
            n_inputs += 512
        if segment_size == 54:
            n_inputs += 768
        if segment_size == 108:
            n_inputs += 1024
        if segment_size == 216:
            n_inputs += 1280

    local_models = []
    for segment_size in segment_size_list:
        loc_model = local_model(segment_size).cuda()
        loc_model.load_state_dict(torch.load('local_model_'+str(segment_size)+'.pt'))
        loc_model.eval()
        local_models.append(loc_model)
    model = global_model(n_inputs, 512).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        nesterov=True
    )
    loss_function = nn.MultiLabelSoftMarginLoss()

    for start in range(0, total_train_size, n_songs):
        print("Loading datasets...", start)
        train_features = np.concatenate(
            [np.load(os.path.join(MTAT_NPY_FOLDER, 'training/'+train_list_pub[i])) for i in range(start, min(start+n_songs, total_train_size))])
        train_labels = np.load(
            os.path.join(MTAT_SPLIT_FOLDER, 'y_train_pub.npy'))[[index[i] for i in range(start, min(start+n_songs, total_train_size))]]
        if normalization:
            mean = np.mean(train_features, axis=0)
            var = np.var(train_features, axis=0)
            train_features = (train_features - mean) / np.sqrt(var)

        train_data = CustomDataset(train_features, train_labels)
        train_size = len(train_data)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        print("Datasets loaded")

        print("Begin training...")
        for e in range(epochs):
            epoch_loss = 0
            correct = 0
            batch_number = 0
            model.train()
            for data, label in train_loader:
                batch_number += 1
                optimizer.zero_grad()
                X = Variable(data[:, :, :1242]).cuda()
                X = torch.cat([loc_model(X)[1] for loc_model in local_models], dim=1)
                Y = Variable(label).cuda().float()
                out, _ = model(X)
                pred = (out.data > 0.50).float()
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
                loss = loss_function(out, Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.data[0]
            total_loss = epoch_loss/batch_number
            train_accuracy = correct/(train_size*number_labels)
            # print("Epoch: {0}, loss: {1:.8f}".format(e+1, total_loss))
            # print("Epoch: {0}, train_accuracy: {1:.8f}".format(e+1, train_accuracy))

    torch.save(model.state_dict(), 'global_model.pt')
    print("Finished training")


def test(segment_size_list):
    test_list_pub = pickle.load(open(os.path.join(MTAT_SPLIT_FOLDER, 'test_list_pub.cP'), 'rb'))
    total_test_size = len(test_list_pub)

    n_inputs = 0
    for segment_size in segment_size_list:
        if segment_size == 18:
            n_inputs += 512
        if segment_size == 27:
            n_inputs += 512
        if segment_size == 54:
            n_inputs += 768
        if segment_size == 108:
            n_inputs += 1024
        if segment_size == 216:
            n_inputs += 1280

    local_models = []
    for segment_size in segment_size_list:
        loc_model = local_model(segment_size).cuda()
        loc_model.load_state_dict(torch.load('local_model_'+str(segment_size)+'.pt'))
        loc_model.eval()
        local_models.append(loc_model)
    model = global_model(n_inputs, 512).cuda()
    model.load_state_dict(torch.load('global_model.pt'))
    model.eval()
    auc = AUCMeter()

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
            out, _ = model(X)
            auc_out = np.reshape(out.data.cpu().numpy(), -1)
            auc_target = np.reshape(labels, -1)
            auc.add(auc_out, auc_target)

    auc_tuple = auc.value()
    print("AUC = ", auc_tuple[0])
    plt.plot(auc_tuple[2], auc_tuple[1])
    plt.plot([0, 1])
    plt.show()


if __name__ == '__main__':
    segment_size_list = []
    for i in range(1, len(sys.argv)):
        segment_size_list.append(int(sys.argv[i]))
    train(segment_size_list)
    test(segment_size_list)
