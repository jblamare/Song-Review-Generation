import torch.nn as nn
from torch.nn import Sequential


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """

    def forward(self, input):
        return input.view(input.size(0), input.size(1))


def cnn_module():
    """
    Create a nn.Sequential model containing all of the layers of the CNN model as specified in the paper.
    https://arxiv.org/pdf/1703.01793.pdf
    """
    return Sequential(
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm1d(num_features=128)
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=3),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm1d(num_features=128)
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=3),
        nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm1d(num_features=256)
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=3),
        nn.Linear(in_features=256, out_features=256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=256, out_features=50),
        nn.Sigmoid()
    )
