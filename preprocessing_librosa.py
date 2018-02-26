import numpy as np
import os
import csv
from multiprocessing.pool import Pool
from time import time

import librosa

from settings import sample_rate, window, n_fft, hop_length, n_mels, n_mfcc, C, DATA_FOLDER

block_size = 2000
offset = 0


def extract_features(audio_path):

    try:
        y, sr = librosa.load(audio_path, sr=sample_rate)
    except EOFError:
        return None
    D = np.abs(librosa.stft(y, window=window, n_fft=n_fft, hop_length=hop_length)) ** 2
    D = np.log(1 + C * D)
    S = librosa.feature.melspectrogram(S=D, y=y, n_mels=n_mels)
    feats = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    return feats


def handle_row(annotation_row):
    labels = np.asarray(annotation_row[1:-1], dtype='int_')
    path = annotation_row[-1]
    features = extract_features(os.path.join(DATA_FOLDER, path))
    return features, labels


if __name__ == '__main__':

    pool = Pool(4)
    features = []
    labels = []
    label_count = np.zeros(188, dtype='int_')


    with open(os.path.join(DATA_FOLDER, 'annotations_randomized.csv')) as annotation_file:

        annotation_reader = csv.reader(annotation_file, delimiter='\t')
        next(annotation_reader)
        for _ in range(offset):
            next(annotation_reader)
        start_time = time()
        print('all set')

        for i, (song_features, song_labels) in enumerate(pool.imap_unordered(handle_row, annotation_reader)):


            if i % block_size == 0 and i > 0:
                labels = np.asarray(labels)
                np.save(os.path.join(DATA_FOLDER, 'magnatagatune_{}_features.npy'.format(int(i / block_size))), features)
                np.save(os.path.join(DATA_FOLDER, 'magnatagatune_{}_labels.npy'.format(int(i / block_size))), labels)
                print('{} songs handled in {} s'.format(i, time() - start_time))
                del features
                del labels
                features = []
                labels = []


            if song_features is None:
                print('empty file detected')
                continue

            features.append(song_features)
            labels.append(song_labels)
            label_count += song_labels

    np.save(os.path.join(DATA_FOLDER, 'label_count.npy'), label_count)
