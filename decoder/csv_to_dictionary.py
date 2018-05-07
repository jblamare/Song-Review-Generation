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
    indexed = [0] + [index_word(character, indexer, reverse_indexer) for character in transcript] + [1]
    return indexed


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
    text = re.sub('^[0-9] \/ [0-9] [Aa]lbums', "", text)
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

    indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'indexer.json')))
    reverse_indexer = json.load(open(os.path.join(REVIEWS_FOLDER, 'reverse_indexer.json')))
    review_dictionary = {}

    json.dump(review_dictionary, open(os.path.join(REVIEWS_FOLDER, 'review_dictionary.json'), 'w'), ensure_ascii=False, indent=2)

    for i, row in enumerate(df.iterrows()):
        artist = row[1]['artist']
        album = row[1]['album']
        review = row[1]['review']

    for i, row in enumerate(df.iterrows()):
        artist = row[1]['artist']
        album = row[1]['album']
        review = row[1]['review']

        if isinstance(review, str) and '*' not in review:
            try:
                review_dictionary[artist][album] = index_transcript(clean_review(review), indexer, reverse_indexer)
            except KeyError:
                review_dictionary[artist] = {'album': index_transcript(clean_review(review), indexer, reverse_indexer)}

        print(i)

        if i == 100:
            json.dump(review_dictionary, open(os.path.join(REVIEWS_FOLDER, 'review_dictionary.json'), 'w'),
                      ensure_ascii=False, indent=2)

    json.dump(review_dictionary, open(os.path.join(REVIEWS_FOLDER, 'review_dictionary.json'), 'w'), ensure_ascii=False, indent=2)

