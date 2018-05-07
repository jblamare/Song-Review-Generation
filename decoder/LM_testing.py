import torch
from torch.autograd import Variable
import numpy as np
import os
import json

from LM_model import LanguageModel
from LM_settings import batch_size, embedding_dim, hidden_dim, music_dim, epochs
from settings import REVIEWS_FOLDER

print('Loading indexer')
# train_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'train_reviews.npy'))
# test_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'test_reviews.npy'))
# dev_reviews = np.load(os.path.join(REVIEWS_FOLDER, 'dev_reviews.npy'))
indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'indexer.json')))
reverse_indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json')))
print('Indexers loaded')

# model = LanguageModel(len(indexer), embedding_dim, hidden_dim)
# model.load_state_dict(
#     torch.load('LanguageModel_1.pt', map_location=lambda storage, loc: storage))
# model.eval()


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def generation(model, text_features, forward, music_features=None, cuda=False):
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
        out = model(x=X, m=music_features)
        out = out[-1, :, :]
        # gumbel = Variable(sample_gumbel(shape=out.size(), out=out.data.new()))
        # out += gumbel
        pred = out.data.max(1, keepdim=True)[1]
        pred = torch.t(pred)
        if i == 0:
            generated = pred
        else:
            generated = torch.cat([generated, pred], dim=0)
        X = torch.cat([X.data, pred], dim=0)
    return torch.t(generated)


def generate_sample(model, forward=30, cuda=False):
    initialization = [0, 2]
    initialization = np.expand_dims(np.array(initialization), axis=0)
    if cuda:
        generated = generation(model, initialization, forward, cuda=cuda).cpu().numpy()[0, :]
    else:
        generated = generation(model, initialization, forward, cuda=cuda).numpy()[0, :]
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
            print(similarity)


if __name__ == "__main__":
    # test_embedding()
    test_generation()
