from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, data, labels):

        self.X, self.Y = data, labels
        self.Y = self.Y.astype(np.int64)

        self.X = self.X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feats = self.X[idx]
        if self.Y is not None:
            labels = self.Y[idx]
            sample = (feats, labels)
        else:
            sample = feats
        return sample
