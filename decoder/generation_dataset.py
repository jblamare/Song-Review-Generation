from torch.utils.data import Dataset
import torch
import numpy as np


class GenerationDataset(Dataset):

    def __init__(self, data, descriptions):

        self.X, self.Y = data, descriptions
        self.X = self.X.astype(np.float32)
        self.Y = self.Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feats = self.X[idx]
        if self.Y is not None:
            labels = np.asarray(self.Y[idx]).astype(np.int64)
            sample = (feats, labels)
        else:
            sample = feats
        return sample


def generation_collate(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    features, descriptions = zip(*data)
    features = torch.from_numpy(np.asarray(features))

    features = torch.stack(features, 0)

    lengths = [len(description) for description in descriptions]
    targets = torch.zeros(len(descriptions), max(lengths)).long()
    for i, description in enumerate(descriptions):
        end = lengths[i]
        targets[i, :end] = torch.from_numpy(description[:end])

    return features, targets, lengths
