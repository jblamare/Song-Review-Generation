import numpy as np
import os

from settings import DATA_FOLDER, number_sets, number_labels


class NPY():

    def __init__(self):
        self.train_set = None
        self.test_set = None

    @property
    def train(self):
        if self.train_set is None:
            features, labels = np.concatenate([load_features(DATA_FOLDER, 'magnatagatune_{}'.format(index + 1)) for index in range(number_sets)]), \
                             np.concatenate([load_labels(DATA_FOLDER, 'magnatagatune_{}'.format(index + 1)) for index in range(number_sets)])
            self.train_set = features, labels
            # mean = np.mean(features, axis=0)
            # var = np.var(features, axis=0)
            # self.train_set = (features - mean) / np.sqrt(var), labels
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(DATA_FOLDER, ''), encoding='bytes'), None)
        return self.test_set


def load_features(path, name):
    return (
        np.load(os.path.join(path, '{}_features.npy'.format(name)), encoding='bytes')
    )

def load_labels(path, name):
    return (
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')[:, best_labels()]
    )

def best_labels():
    label_count = np.load(os.path.join(DATA_FOLDER, 'label_count.npy'))
    indices = np.argpartition(label_count, -number_labels)[-number_labels:]
    return indices


if __name__ == '__main__':
    label_count = np.load(os.path.join(DATA_FOLDER, 'label_count.npy'))
    print(best_labels())
    print(label_count[best_labels()])