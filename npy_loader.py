import numpy as np
import os
from settings import DATA_FOLDER


class NPY():

    def __init__(self):
        self.train_set = None
        self.test_set = None

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(DATA_FOLDER, 'magnatagatune_1')
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(
                DATA_FOLDER, ''), encoding='bytes'), None)
        return self.test_set


def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}_features.npy'.format(name)), encoding='bytes'),
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )
