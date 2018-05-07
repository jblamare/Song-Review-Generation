import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
from LM_model import LanguageModel
from LM_loader import MyLoader
from LM_settings import embedding_dim, hidden_dim, epochs, REVIEWS_FOLDER
import json


class Trainer():
    """
    A simple training cradle
    """

    def __init__(self, vocab_size, model, optimizer, train_data):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.vocab_size = vocab_size
        self.loss_function = nn.CrossEntropyLoss()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def init_weight(self, m):
        if type(m) == nn.Embedding:
            m.weight.data.uniform_(-0.1, 0.1)
        if type(m) == nn.LSTM:
            m.weight_ih_l0.data.uniform_(-1/np.sqrt(hidden_dim),
                                         1/np.sqrt(hidden_dim))
            m.weight_hh_l0.data.uniform_(-1/np.sqrt(hidden_dim),
                                         1/np.sqrt(hidden_dim))
            m.bias_ih_l0.data.fill_(0)
            m.bias_hh_l0.data.fill_(0)
        if type(m) == nn.Linear:
            smoothing = 0.1
            unigram = np.zeros(self.vocab_size)
            for article in self.train_data:
                for word in article:
                    unigram[word] += 1
            unigram = np.log(unigram/self.vocab_size)
            unigram = unigram*(1-smoothing)+smoothing/self.vocab_size
            unigram = torch.from_numpy(unigram)
            m.bias.data.copy_(unigram)

    def run(self, epochs):
        print("Begin training...")
        train_loader = MyLoader(self.train_data)
        for e in range(epochs):
            epoch_loss = 0
            batch_number = 0
            self.model.train()
            for in_data, out_data in train_loader:
                batch_number += 1
                self.optimizer.zero_grad()
                X = Variable(torch.from_numpy(in_data).long()).cuda()
                Y = Variable(torch.from_numpy(out_data).long()).cuda()
                out = self.model(X)
                out = out.view(out.shape[0]*out.shape[1], out.shape[2])
                Y = Y.view(-1)
                loss = self.loss_function(out, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data[0]
            total_loss = epoch_loss/batch_number
            print("Epoch: {0}, train_loss: {1:.8f}".format(e+1, total_loss))
            self.save_model('LanguageModel_'+str(e)+'.pt')


def train():

    print('Loading datasets')
    train_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'train_reviews.npy'))
    test_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'test_reviews.npy'))
    dev_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'dev_reviews.npy'))
    indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'indexer.json')))
    reverse_indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json')))
    reviews = np.concatenate((train_reviews, dev_reviews, test_reviews))
    del train_reviews, dev_reviews, test_reviews
    print('Datasets loaded')

    # TRAIN MODELS
    vocab_size = len(indexer)
    model = LanguageModel(vocab_size, embedding_dim, hidden_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(vocab_size, model, optimizer, reviews)
    model.apply(trainer.init_weight)
    trainer.run(epochs)
    trainer.save_model('LanguageModel.pt')


if __name__ == "__main__":
    train()
