from torch.utils.data import Dataset
from torch import Tensor
from npy_loader import NPY
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, type):

        loader = NPY()

        if type == 'train':
            self.X, self.Y = loader.train(True)
            self.Y = self.Y.astype(np.int64)
        if type == 'test':
            self.X, self.Y = loader.test(True)

        self.X = self.X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feats = self.X[idx]
        if self.Y is not None:
            label = self.Y[idx]
            sample = (feats, label)
        else:
            sample = feats
        return sample
