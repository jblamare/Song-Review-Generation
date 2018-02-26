import numpy as np
import os


class NPY():
    """ Load the NPY dataset

        Ensure NPY_PATH is path to directory containing
        all data files (.npy) provided on Kaggle.

        Example usage:
            loader = NPY()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)

    """

    def __init__(self):
        self.train_set = None
        self.test_set = None

    #@property
    def train(self, preprocessed):
        if preprocessed:
            name = 'train_prep'
        else:
            name = 'train'
        if self.train_set is None:
            self.train_set = load_raw(os.environ['NPY_PATH'], name)
        return self.train_set

    #@property
    def test(self, preprocessed):
        if preprocessed:
            name = 'test_prep_feats.npy'
        else:
            name = 'test_feats.npy'
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(
                os.environ['NPY_PATH'], name), encoding='bytes'), None)
        return self.test_set


def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}_feats.npy'.format(name)), encoding='bytes'),
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )
