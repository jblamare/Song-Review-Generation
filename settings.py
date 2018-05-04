import os

# preprocessing
n_mels = 40
n_mfcc = 128
sample_rate = 22050
n_fft = 1024
hop_length = 512
window = "hann"
C = 10

train_dataset = 'MTAT'  # 'MSD'
test_dataset = 'MTAT'  # 'MSD'

# data loading JB

DATA_FOLDER = "/media/jblamare/My Passport/SongReviewGeneration/Full dataset/"
MTAT_MP3_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MTAT/mp3/"
MTAT_NPY_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MTAT/npy/"
MTAT_SPLIT_FOLDER = "/home/jblamare/Documents/CMU/11-747/Project/music_dataset_split/MTAT_split/"
MSD_DATABASE_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MillionSongDataset/AdditionalFiles/"
MSD_SONGS_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MillionSongDataset/"
MSD_MP3_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/songs/"
MSD_NPY_FOLDER = "/media/jblamare/SAMSUNG/npy/"
MSD_SPLIT_FOLDER = "/home/jblamare/Documents/CMU/11-747/Project/music_dataset_split/MSD_split/"
MSD_CODE_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MSongsDB"

# data loading TVN

# TVN_FOLDER = "/home/teven/Projet747/"
# DATA_FOLDER = TVN_FOLDER + "Full dataset/"
# MTAT_MP3_FOLDER = TVN_FOLDER + "MusicDatasets/MTAT/mp3/"
# MTAT_NPY_FOLDER = TVN_FOLDER + "MusicDatasets/MTAT/npy/"
# MTAT_SPLIT_FOLDER = TVN_FOLDER + "music_dataset_split/MTAT_split/"
# MTAT_GENERATION_SPLIT = TVN_FOLDER + 'music_dataset_split/MTAT_generation_split'
# MSD_DATABASE_FOLDER = TVN_FOLDER + "MusicDatasets/MSD/MillionSongDataset/AdditionalFiles/"
# MSD_SONGS_FOLDER = TVN_FOLDER + "MusicDatasets/MSD/MillionSongDataset/"
# MSD_SPLIT_FOLDER = TVN_FOLDER + "music_dataset_split/MSD_split/"
# CLIP_INFO_FILE = TVN_FOLDER + "clip_info_final.csv"


# DATA_FOLDER = os.path.join('../mp3/')

number_sets = 1
number_labels = 50
n_songs = 2000
normalization = False

# model training

seed = 43
batch_size = 64
epochs = 10
learning_rate = 0.1  # 0.0005 (Adam)
momentum = 0.9
