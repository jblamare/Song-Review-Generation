import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
import json
from LM_model import LanguageModel
from LM_loader import MyLoader
from settings import REVIEWS_FOLDER, MSD_SPLIT_FOLDER, MSD_NPY_FOLDER, PITCHFORK_DB_PATH
from LM_settings import batch_size, embedding_dim, hidden_dim, music_dim, epochs
from LM_testing import generate_sample
from review_extractor import index_transcript, clean_review
from model import local_model, global_model
import pickle
import pandas
import sqlite3

from time import time


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
        step = 10000
        # print(len(train_loader))

        for e in range(epochs):

            start_time = time()
            epoch_loss = 0
            batch_number = 0
            self.model.train()
            current_threshold = step

            for i, (in_data, out_data) in enumerate(train_loader):
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

                # print(time() - start_time)

                if i == current_threshold:
                    print("{0} examples treated in {1:.4f}s".format(i, time() - start_time))
                    current_threshold += step

            total_loss = epoch_loss/batch_number
            print("Epoch: {0}, train_loss: {1:.8f}, time: {2:.4f}".format(
                e+1, total_loss, time() - start_time))
            generate_sample(self.model, cuda=True)
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
    model = LanguageModel(vocab_size, embedding_dim, hidden_dim, music_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(vocab_size, model, optimizer, reviews)
    model.apply(trainer.init_weight)
    trainer.run(epochs)
    trainer.save_model('LanguageModel.pt')


def train_with_audio():
    indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'indexer.json')))
    reverse_indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json')))
    vocab_size = len(indexer)

    language_model = LanguageModel(vocab_size, embedding_dim, hidden_dim, music_dim).cuda()
    language_model.load_state_dict(torch.load('LanguageModel_4.pt'))
    language_model.train()
    # filter(lambda p: p.requires_grad, language_model.parameters()))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, language_model.parameters()))
    loss_function = nn.CrossEntropyLoss()

    n_inputs = 512+512+768
    segment_size_list = [18, 27, 54]
    local_models = []
    for segment_size in segment_size_list:
        loc_model = local_model(segment_size).cuda()
        loc_model.load_state_dict(torch.load('local_model_'+str(segment_size)+'.pt'))
        loc_model.eval()
        local_models.append(loc_model)
    model = global_model(n_inputs, 512).cuda()
    model.load_state_dict(torch.load('global_model_18_27_54_9051_123.pt'))
    model.eval()

    id7d_to_path = pickle.load(open(os.path.join(MSD_SPLIT_FOLDER, '7D_id_to_path.pkl'), 'rb'))
    idmsd_to_id7d = pickle.load(open(os.path.join(MSD_SPLIT_FOLDER, 'MSD_id_to_7D_id.pkl'), 'rb'))

    pairs = json.load(open('pairs.json'))

    for e in range(epochs):

        start_time = time()
        epoch_loss = 0
        batch_number = 0
        song_number = 0

        for track_id, value in pairs.items():
            song_number += 1

            if song_number < 6:
                print("for testing")
                continue

            review = value['review']
            review = np.expand_dims(np.array(review), axis=0)

            try:
                npy_path = os.path.join(
                    MSD_NPY_FOLDER+'new_pitchfork', id7d_to_path[idmsd_to_id7d[track_id]][:-9]+'.npy')
                X = np.load(npy_path)
            except KeyError:
                print("No key?")
                continue
            except FileNotFoundError:
                print("No audio?")
                continue

            try:
                X = torch.from_numpy(X[:, :1255]).unsqueeze(0)
                X = Variable(X).cuda().float()
                X = torch.cat([loc_model(X)[1] for loc_model in local_models], dim=1)
                _, music = model(X)
            except:
                print("Weird song (too short?)")
                print(X.shape)
                continue

            train_loader = MyLoader(review)
            for in_data, out_data in train_loader:
                batch_number += 1
                optimizer.zero_grad()
                X = Variable(torch.from_numpy(in_data).long()).cuda()
                Y = Variable(torch.from_numpy(out_data).long()).cuda()
                expanded_music = music.unsqueeze(1).expand(X.shape[0], X.shape[1], -1).contiguous()
                out = language_model(X, expanded_music)
                out = out.view(out.shape[0]*out.shape[1], out.shape[2])
                Y = Y.view(-1)
                loss = loss_function(out, Y)
                loss.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss.data.item()

        total_loss = epoch_loss/batch_number
        print("Epoch: {0}, train_loss: {1:.8f}, time: {2:.4f}".format(
            e+1, total_loss, time() - start_time))
        generate_sample(language_model, cuda=True)
        torch.save(language_model.state_dict(), 'LanguageModel_audio_'+str(e)+'.pt')

    conn_pf.close()


if __name__ == "__main__":
    # train()
    train_with_audio()
