"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from npy_loader import NPY
from all_cnn import cnn_module
from custom_dataset import CustomDataset


# def write_results(predictions, output_file='predictions.txt'):
#     """
#     Write predictions to file for submission.
#     File should be:
#         named 'predictions.txt'
#         in the root of your tar file
#     :param predictions: iterable of integers
#     :param output_file:  path to output file.
#     :return: None
#     """
#     with open(output_file, 'w') as f:
#         for y in predictions:
#             f.write("{}\n".format(int(y)))


def preprocess():
    loader = NPY()
    X_train, Y_train = loader.train(False)
    X_test, _ = loader.test(False)
    print("Starting preprocessing...")
    X_train, X_test = cifar_10_preprocess(X_train, X_test)
    print("Done")
    np.save("dataset/train_prep_feats.npy", X_train)
    np.save("dataset/train_prep_labels.npy", Y_train)
    np.save("dataset/test_prep_feats.npy", X_test)


def init_xavier(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0)


def main():
    # Hyperparameters
    batch_size = 32
    epochs = 20
    learning_rate = 0.01
    momentum = 0.9
    regularization = 0.001

    print("Loading datasets...")
    train_data = CustomDataset('train')
    test_data = CustomDataset('test')
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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, nesterov=True)
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
            pred = out.data.max(1, keepdim=True)[1]
            predicted = pred.eq(Y.data.view_as(pred))
            correct += predicted.sum()
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(out, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
        total_loss = epoch_loss/batch_number
        train_accuracy = correct/train_size
        val_accuracy = inference(model, val_loader, val_size)
        print("Epoch: {0}, loss: {1:.8f}".format(e+1, total_loss))
        print("Epoch: {0}, train_accuracy: {1:.8f}".format(e+1, train_accuracy))
        print("Epoch: {0}, val_accuracy: {1:.8f}".format(e+1, val_accuracy))
    print("Finished training")

    # print("Begin testing...")
    # y = np.array([])
    # for data in test_loader:
    #     X = Variable(data).cuda()
    #     out = model(X)
    #     pred = out.data.max(1, keepdim=True)[1]
    #     pred = pred.cpu().numpy().reshape(-1,)
    #     y = np.concatenate((y, pred))
    # write_results(y)
    # print("Finished testing")


if __name__ == '__main__':
    # preprocess()
    main()
