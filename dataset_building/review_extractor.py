import sqlite3
import os
import re
from unicodedata import normalize
import random
from settings import PITCHFORK_CSV_PATH, REVIEWS_FOLDER
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
import json


def index_transcript(transcript, indexer, reverse_indexer):
    indexed = np.asarray([0] + [index_word(character, indexer, reverse_indexer) for character in transcript] + [1])
    return np.asarray(indexed)


def index_word(word, indexer, reverse_indexer):
    try:
        return indexer[word]
    except KeyError:
        new_index = len(indexer)
        indexer[word] = new_index
        reverse_indexer.append(word)
        return new_index


def clean_review(text):
    text = normalize('NFKD', text)
    text = re.sub("-", " - ", text)
    text = re.sub("—", " — ", text)
    text = re.sub("/", " / ", text)
    text = re.sub("\s\s+", " ", text)
    text = re.sub('^Best new [A-Za-z]*', "", text)
    text = text.strip()
    text = re.sub('^[0-9] / [0-9] [Aa]lbums', "", text)
    text = text.strip()
    # text = text.replace("\\n\\", "\n")
    start_dot_matches = re.finditer("\.\.\.", text)
    for start_dots in start_dot_matches:
        if start_dots is not None:
            if text[start_dots.start() + 4:start_dots.start() + 14] == text[:10]:
                text = text[start_dots.start() + 4:]
    sentences = [sentence for sentence in sent_tokenize(text)]
    sentences = [word_tokenize(sentence) for sentence in sentences]
    return [word.lower() for sentence in sentences for word in sentence]


def indexes_to_characters(transcript, dictionary):
    text = " ".join([dictionary[index] for index in transcript[1:-1]])
    return text


if __name__ == '__main__':

    df = pd.read_csv(open(PITCHFORK_CSV_PATH, encoding='cp1252'))
    indexed_reviews = []
    unclean_reviews = []

    indexer = {'<sos>': 0, '<eos>': 1}
    reverse_indexer = ['<sos>', '<eos>']

    for i, review in enumerate(df['review']):
        if isinstance(review, str) and '*' not in review:
            unclean_reviews.append(review)
            indexed_reviews.append(index_transcript(clean_review(review), indexer, reverse_indexer))

    random.Random(1793).shuffle(indexed_reviews)
    indexed_reviews = np.asarray(indexed_reviews)

    print(unclean_reviews[-200])

    train_reviews = indexed_reviews[:int(i*0.8)]
    test_reviews = indexed_reviews[int(i*0.8):int(i*0.9)]
    dev_reviews = indexed_reviews[int(i*0.9):]

    json.dump(indexer, open(os.path.join(REVIEWS_FOLDER, 'indexer.json'), 'w'), ensure_ascii=False, indent=2)
    json.dump(reverse_indexer, open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json'), 'w'), ensure_ascii=False, indent=2)
    np.save(os.path.join(REVIEWS_FOLDER, 'train_reviews.npy'), train_reviews)
    np.save(os.path.join(REVIEWS_FOLDER, 'test_reviews.npy'), test_reviews)
    np.save(os.path.join(REVIEWS_FOLDER, 'dev_reviews.npy'), dev_reviews)

    print("{} words found".format(len(indexer)))
