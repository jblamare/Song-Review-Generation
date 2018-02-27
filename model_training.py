import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from npy_loader import NPY
from model import cnn_module
from custom_dataset import CustomDataset
from torchnet.meter import AUCMeter
import matplotlib.pyplot as plt
from settings import number_labels


def init_xavier(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0)


def inference(model, loader, n_members):
    model.eval()
    correct = 0
    for data, labels in loader:
        X = Variable(data).cuda()
        Y = Variable(labels).cuda()
        out = model(X)
        pred = (out.data > 0.50).long()
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
    return correct / n_members


def inference_auc(model, loader):
    model.eval()
    auc = AUCMeter()
    for data, labels in loader:
        X = Variable(data).cuda()
        out = model(X)
        auc_out = np.reshape(out.data.cpu().numpy(), -1)
        auc_target = np.reshape(labels, -1)
        auc.add(auc_out, auc_target)
    auc_tuple = auc.value()
    print("AUC = ", auc_tuple[0])
    plt.plot(auc_tuple[2], auc_tuple[1])
    plt.plot([0, 1])
    plt.show()


def main():
    # Hyperparameters
    segment_size = 27
    batch_size = 16
    epochs = 5
    learning_rate = 0.0000000001
    momentum = 0.9
    regularization = 0.01

    print("Loading datasets...")
    train_data = CustomDataset('train', segment_size)
    train_size = len(train_data)

    val_size, train_size = int(0.20 * train_size), int(0.80 * train_size)  # 80 / 20 train-val split

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(val_size, val_size+train_size)))
    val_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(0, val_size)))
    print("Datasets loaded")

    print("Begin training...")
    model = cnn_module().cuda()
    model.apply(init_xavier)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for e in range(epochs):
        epoch_loss = 0
        correct = 0
        batch_number = 0
        for data, label in train_loader:
            batch_number += 1
            optimizer.zero_grad()
            X = Variable(data).cuda()
            Y = Variable(label).cuda()
            out = model(X)
            pred = (out.data > 0.50).long()
            if e == epochs - 1:
                print(out)
                print(pred)
            predicted = pred.eq(Y.data.view_as(pred))
            correct += predicted.sum()
            loss_function = nn.MultiLabelMarginLoss()
            loss = loss_function(out, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
        total_loss = epoch_loss/batch_number
        train_accuracy = correct/(train_size*number_labels)
        val_accuracy = inference(model, val_loader, (val_size*number_labels))
        print("Epoch: {0}, loss: {1:.8f}".format(e+1, total_loss))
        print("Epoch: {0}, train_accuracy: {1:.8f}".format(e+1, train_accuracy))
        print("Epoch: {0}, val_accuracy: {1:.8f}".format(e+1, val_accuracy))

    inference_auc(model, val_loader)
    print("Finished training")


if __name__ == '__main__':
    main()
