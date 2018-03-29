from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, data, labels):  # , segment_size):

        # self.segment_size = segment_size

        self.X, self.Y = data, labels
        self.Y = self.Y.astype(np.int64)

        self.X = self.X.astype(np.float32)
        # self.n_frames = self.X[0].shape[1]
        # self.n_segments = self.n_frames // self.segment_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feats = self.X[idx]  # [:, :self.segment_size*self.n_segments]
        if self.Y is not None:
            labels = self.Y[idx]
            sample = (feats, labels)
        else:
            sample = feats
        return sample
