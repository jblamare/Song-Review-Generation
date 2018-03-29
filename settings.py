import os

# preprocessing
n_mels = 40
n_mfcc = 128
sample_rate = 22050
n_fft = 1024
hop_length = 512
window = "hann"
C = 10

# data loading
DATA_FOLDER = "/media/jblamare/My Passport/SongReviewGeneration/Full dataset/"
MTAT_MP3_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MTAT/mp3/"
MTAT_NPY_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MTAT/npy/"
MTAT_SPLIT_FOLDER = "/home/jblamare/Documents/CMU/11-747/Project/music_dataset_split/MTAT_split/"
MSD_DATABASE_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MillionSongDataset/AdditionalFiles/"
MSD_SONGS_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MillionSongDataset/"
MSD_SPLIT_FOLDER = "/home/jblamare/Documents/CMU/11-747/Project/music_dataset_split/MSD_split/"
# DATA_FOLDER = os.path.join('../mp3/')
number_sets = 1
number_labels = 50
n_songs = 2048
normalization = True

# model training
seed = 43
batch_size = 64
epochs = 10
learning_rate = 0.1  # 0.0005 (Adam)
momentum = 0.9
