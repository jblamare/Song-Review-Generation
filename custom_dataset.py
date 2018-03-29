from torch.utils.data import Dataset
from npy_loader import NPY
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, type, segment_size, first):

        loader = NPY()
        self.segment_size = segment_size

        if type == 'train':
            self.X, self.Y = loader.train(first)
            self.Y = self.Y.astype(np.int64)
        # if type == 'test':
        #     self.X, self.Y = loader.test

        self.X = self.X.astype(np.float32)
        self.n_frames = self.X[0].shape[1]
        self.n_segments = self.n_frames // self.segment_size

    def __len__(self):
        # return len(self.X) * self.n_segments
        return len(self.X)

    def __getitem__(self, idx):
        # song_id = idx // self.n_segments
        # segment_id = idx % self.n_segments
        # feats = self.X[song_id][:, segment_id*self.segment_size: (segment_id+1)*self.segment_size]
        feats = self.X[idx][:, :self.segment_size*self.n_segments]
        if self.Y is not None:
            labels = self.Y[idx]
            sample = (feats, labels)
        else:
            sample = feats
        return sample


if __name__ == "__main__":
    train_data = CustomDataset('train', 27, 1)
    print(len(train_data))
    print(train_data[0][0].shape)
    print(train_data[0][1].shape)
