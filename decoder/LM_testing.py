import torch
from torch.autograd import Variable
import numpy as np
import os
import json
import pickle

from LM_model import LanguageModel
from LM_settings import batch_size, embedding_dim, hidden_dim, music_dim, epochs, music_dim
from settings import REVIEWS_FOLDER, MSD_NPY_FOLDER, MSD_SPLIT_FOLDER
from model import local_model, global_model


print('Loading indexer')
# train_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'train_reviews.npy'))
# test_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'test_reviews.npy'))
# dev_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'dev_reviews.npy'))
indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'indexer.json')))
reverse_indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json')))
vocab_size = len(indexer)
print('Indexers loaded')

id7d_to_path = pickle.load(open(os.path.join(MSD_SPLIT_FOLDER, '7D_id_to_path.pkl'), 'rb'))
idmsd_to_id7d = pickle.load(open(os.path.join(MSD_SPLIT_FOLDER, 'MSD_id_to_7D_id.pkl'), 'rb'))


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def generation(model, text_features, forward, with_gumbel, music_features=None, cuda=False):
    """
    Generate a sequence of words given a starting sequence.
    Load your model before generating words.
    :param text_features: Initial sequence of words (batch size, length)
    :param forward: number of additional words to generate
    :return: generated words (batch size, forward)
    """
    X = torch.from_numpy(np.transpose(text_features)).long()
    for i in range(forward):
        X = Variable(X)
        if cuda:
            X = X.cuda()
            if music_features is not None:
                music_features = music_features.cuda()

        if music_features is not None:
            expanded_music = music_features.unsqueeze(1).expand(
                X.shape[0], X.shape[1], -1).contiguous()
        out = model(x=X, m=expanded_music)
        out = out[-1, :, :]
        if with_gumbel:
            gumbel = Variable(sample_gumbel(shape=out.size(), out=out.data.new()))
            out += gumbel
        pred = out.data.max(1, keepdim=True)[1]
        pred = torch.t(pred)
        if i == 0:
            generated = pred
        else:
            generated = torch.cat([generated, pred], dim=0)
        X = torch.cat([X.data, pred], dim=0)
    return torch.t(generated)


def generate_sample(language_model, forward=100, cuda=False, with_gumbel=False):
    pairs = json.load(open('pairs.json'))
    song_number = 0

    n_inputs = 512+512+768
    segment_size_list = [18, 27, 54]
    local_models = []
    for segment_size in segment_size_list:
        loc_model = local_model(segment_size)
        if cuda:
            loc_model = loc_model.cuda()
        loc_model.load_state_dict(torch.load('local_model_'+str(segment_size)+'.pt'))
        loc_model.eval()
        local_models.append(loc_model)
    model = global_model(n_inputs, 512)
    if cuda:
        model = model.cuda()
    model.load_state_dict(torch.load('global_model_18_27_54_9051_123.pt'))
    model.eval()

    for track_id, value in pairs.items():
        song_number += 1
        if song_number >= 50:
            break
        else:
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
                X = Variable(X).float()
                if cuda:
                    X = X.cuda()
                X = torch.cat([loc_model(X)[1] for loc_model in local_models], dim=1)
                _, music = model(X)
            except:
                print("Weird song (too short?)")
                print(X.shape)
                continue

            initialization = [0]
            initialization = np.expand_dims(np.array(initialization), axis=0)
            if cuda:
                generated = generation(language_model, initialization, forward,
                                       music_features=music, cuda=cuda, with_gumbel=with_gumbel).cpu().numpy()[0, :]
            else:
                generated = generation(language_model, initialization, forward,
                                       music_features=music, cuda=cuda, with_gumbel=with_gumbel).numpy()[0, :]
            print('----------------')
            print(value['title'])
            print(value['album'])
            print(value['artist'])
            # print(' '.join([reverse_indexer[i] for i in value['review']]))
            print(' '.join([reverse_indexer[i] for i in generated]))


def test_generation():
    initialization = []
    while(True):
        word = input("next word: ")
        if word == "<q>":
            break
        elif word not in indexer:
            print("Not in indexer")
            continue
        else:
            initialization.append(indexer[word])
    initialization = np.expand_dims(np.array(initialization), axis=0)
    forward = int(input("Forward: "))
    generated = generation(initialization, forward).numpy()[0, :]
    print(' '.join([reverse_indexer[i] for i in generated]))


def test_embedding(model):
    embedding_weights = model.get_embedding()
    while(True):
        word1 = input("First word for similarity: ")
        if word1 == "<q>":
            break
        elif word1 not in indexer:
            print("Not in indexer")
            continue
        else:
            vector1 = embedding_weights[indexer[word1]]
        word2 = input("Second word for similarity: ")
        if word2 == "<best>":
            best_score = 0
            best_word = ''
            for word in indexer:
                if word != word1:
                    vector2 = embedding_weights[indexer[word]]
                    similarity = torch.dot(vector1, vector2) / \
                        (torch.norm(vector1) * torch.norm(vector2))
                    similarity = similarity.item()
                    if best_score == 0 or similarity > best_score:
                        best_score = similarity
                        best_word = word
            print(best_word, best_score)
        elif word2 not in indexer:
            print("Not in indexer")
            continue
        else:
            vector2 = embedding_weights[indexer[word2]]
            similarity = torch.dot(vector1, vector2) / (torch.norm(vector1) * torch.norm(vector2))
            similarity = similarity.item()
            print(similarity)


if __name__ == "__main__":
    language_model = LanguageModel(vocab_size, embedding_dim, hidden_dim, music_dim).cuda()
    language_model.load_state_dict(
        torch.load('LanguageModel_audio_4.pt'))  # , map_location=lambda storage, loc: storage))
    language_model.eval()
    # test_embedding(language_model)
    # test_generation()
    generate_sample(language_model, cuda=True, with_gumbel=True)
