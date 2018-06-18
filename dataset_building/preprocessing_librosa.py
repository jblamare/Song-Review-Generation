import numpy as np
import os
import csv
from multiprocessing.pool import Pool
from time import time
import pickle
import librosa
import sys
import pandas
import audioread
import json

# import librosa.display
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()


from settings import sample_rate, window, n_fft, hop_length, n_mels, n_mfcc, C, DATA_FOLDER, MSD_MP3_FOLDER, MSD_NPY_FOLDER, MSD_SPLIT_FOLDER

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


if __name__ == "__main__":
    train_list_pub = pickle.load(
        open(os.path.join(MSD_SPLIT_FOLDER, 'filtered_list_train.cP'), 'rb'))
    valid_list_pub = train_list_pub[201680:]
    train_list_pub = train_list_pub[0:201680]
    test_list_pub = pickle.load(
        open(os.path.join(MSD_SPLIT_FOLDER, 'filtered_list_test.cP'), 'rb'))

    id7d_to_path = pickle.load(open(os.path.join(MSD_SPLIT_FOLDER, '7D_id_to_path.pkl'), 'rb'))
    idmsd_to_id7d = pickle.load(open(os.path.join(MSD_SPLIT_FOLDER, 'MSD_id_to_7D_id.pkl'), 'rb'))

    if sys.argv[1] == 'train':
        i = 68000
        for song in train_list_pub[68000:]:
            if i % 1000 == 0:
                print(i)
            mp3_path = id7d_to_path[idmsd_to_id7d[song]]
            npy_path = 'training/'+mp3_path[:-9]+'.npy'
            npy_path = os.path.join(MSD_NPY_FOLDER, npy_path)
            try:
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                feats = extract_features(os.path.join(MSD_MP3_FOLDER, mp3_path))
                np.save(npy_path, feats)
            except audioread.NoBackendError:
                print("BackendError")
            except KeyError:
                print("KeyError")
            i += 1

    if sys.argv[1] == 'test':
        i = 0
        for song in test_list_pub:
            if i % 1000 == 0:
                print(i)
            try:
                mp3_path = id7d_to_path[idmsd_to_id7d[song]]
                npy_path = 'testing/'+mp3_path[:-9]+'.npy'
                npy_path = os.path.join(MSD_NPY_FOLDER, npy_path)
                # if os.path.isfile(npy_path):
                #     print(npy_path)
                #     os.remove(npy_path)
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                feats = extract_features(os.path.join(MSD_MP3_FOLDER, mp3_path))
                np.save(npy_path, feats)
            except audioread.NoBackendError:
                print("BackendError")
            except KeyError:
                print("KeyError")
            i += 1

    if sys.argv[1] == 'valid':
        i = 0
        for song in valid_list_pub:
            if i % 1000 == 0:
                print(i)
            try:
                mp3_path = id7d_to_path[idmsd_to_id7d[song]]
                npy_path = 'validation/'+mp3_path[:-9]+'.npy'
                npy_path = os.path.join(MSD_NPY_FOLDER, npy_path)
                # if os.path.isfile(npy_path):
                #     print(npy_path)
                #     os.remove(npy_path)
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                feats = extract_features(os.path.join(MSD_MP3_FOLDER, mp3_path))
                np.save(npy_path, feats)
            except audioread.NoBackendError:
                print("BackendError")
            except KeyError:
                print("KeyError")
            i += 1

    if sys.argv[1] == 'pitchfork':
        pairs = json.load(open(os.path.join(MSD_SPLIT_FOLDER, 'pairs.json')))
        # pairing = pandas.read_csv('pitchfork_msd_long.csv', sep=',', header=0).iloc[1000:]
        index = 0
        for track_id, _ in pairs.items():
            if index % 1000 == 0:
                print(index)
            index += 1
            try:
                mp3_path = id7d_to_path[idmsd_to_id7d[track_id]]
                npy_path = 'new_pitchfork/'+mp3_path[:-9]+'.npy'
                npy_path = os.path.join(MSD_NPY_FOLDER, npy_path)
                # if os.path.isfile(npy_path):
                #     print(npy_path)
                #     os.remove(npy_path)
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                feats = extract_features(os.path.join(MSD_MP3_FOLDER, mp3_path))
                np.save(npy_path, feats)
            except audioread.NoBackendError:
                print("BackendError")
            except KeyError:
                print("KeyError")


# if __name__ == '__main__':
#
#     pool = Pool(4)
#     features = []
#     labels = []
#     label_count = np.zeros(188, dtype='int_')
#
#     with open(os.path.join(DATA_FOLDER, 'annotations_randomized.csv')) as annotation_file:
#
#         annotation_reader = csv.reader(annotation_file, delimiter='\t')
#         next(annotation_reader)
#         for _ in range(offset):
#             next(annotation_reader)
#         start_time = time()
#         print('all set')
#
#         for i, (song_features, song_labels) in enumerate(pool.imap_unordered(handle_row, annotation_reader)):
#
#             if i % block_size == 0 and i > 0:
#                 labels = np.asarray(labels)
#                 np.save(os.path.join(DATA_FOLDER, 'magnatagatune_{}_features.npy'.format(
#                     int(i / block_size))), features)
#                 np.save(os.path.join(DATA_FOLDER, 'magnatagatune_{}_labels.npy'.format(
#                     int(i / block_size))), labels)
#                 print('{} songs handled in {} s'.format(i, time() - start_time))
#                 del features
#                 del labels
#                 features = []
#                 labels = []
#
#             if song_features is None:
#                 print('empty file detected')
#                 continue
#
#             features.append(song_features)
#             labels.append(song_labels)
#             label_count += song_labels
#
#     np.save(os.path.join(DATA_FOLDER, 'label_count.npy'), label_count)
