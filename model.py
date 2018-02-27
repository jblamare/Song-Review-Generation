import torch.nn as nn
from torch.nn import Sequential
from settings import number_labels


class GetSize(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


class Reshape(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], input.shape[2], input.shape[1])


class Flatten(nn.Module):
    def forward(self, input):
        return input.squeeze(2)


def cnn_module():
    """
    Create a nn.Sequential model containing all of the layers of the CNN model as specified in the paper.
    https://arxiv.org/pdf/1703.01793.pdf
    """
    return Sequential(
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        # nn.BatchNorm1d(num_features=128),
        nn.Softplus(),
        nn.AvgPool1d(kernel_size=3, stride=3),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        # nn.BatchNorm1d(num_features=128),
        nn.Softplus(),
        nn.AvgPool1d(kernel_size=414),
        Flatten(),
        nn.Linear(in_features=128, out_features=number_labels),
        nn.Sigmoid()
    )
