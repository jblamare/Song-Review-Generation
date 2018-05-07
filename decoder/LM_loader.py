from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
from random import shuffle, randint
from LM_settings import batch_size


class MyLoader(DataLoader):
    def __init__(self, review_list):
        # overridden because we don't need the super constructor
        self.review_list = review_list

    def __iter__(self):
        base_seq_length = 70
        if(randint(1, 100) > 95):
            base_seq_length = 35

        shuffle(self.review_list)
        data = np.concatenate([review for review in self.review_list])
        M = int(len(data)/batch_size)
        data = data[:batch_size*M+1]
        inputs = data[:-1].reshape((batch_size, M)).T
        outputs = data[1:].reshape((batch_size, M)).T

        pos = 0
        seq_length = int(np.random.normal(loc=base_seq_length, scale=5))
        while M-1-pos > seq_length:
            yield inputs[pos:pos+seq_length], outputs[pos:pos+seq_length]
            pos += seq_length
            seq_length = int(np.random.normal(loc=base_seq_length, scale=5))
