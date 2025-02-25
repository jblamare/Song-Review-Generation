import os

# preprocessing
n_mels = 40
n_mfcc = 128
sample_rate = 22050
n_fft = 1024
hop_length = 512
window = "hann"
C = 10

train_dataset = 'MSD'  # 'MSD'/'MTAT'
test_dataset = 'MSD'  # 'MSD'/'MTAT'

# data loading JB

# DATA_FOLDER = "/media/jblamare/My Passport/SongReviewGeneration/Full dataset/"
# MTAT_MP3_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MTAT/mp3/"
# MTAT_NPY_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MTAT/npy/"
# MTAT_SPLIT_FOLDER = "/home/jblamare/Documents/CMU/11-747/Project/music_dataset_split/MTAT_split/"
# MSD_DATABASE_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MillionSongDataset/AdditionalFiles/"
# MSD_SONGS_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MillionSongDataset/"
# MSD_MP3_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/songs/"
# MSD_NPY_FOLDER = "/media/jblamare/SAMSUNG/npy/"
# MSD_SPLIT_FOLDER = "/home/jblamare/Documents/CMU/11-747/Project/music_dataset_split/MSD_split/"
# MSD_CODE_FOLDER = "/media/jblamare/My Passport/MusicDatasets/MSD/MSongsDB"
# PITCHFORK_DB_PATH = "/media/jblamare/My Passport/MusicDatasets/MSD/database.sqlite"
# PITCHFORK_CSV_PATH = "/media/jblamare/SAMSUNG/Pitchfork/p4kreviews.csv"
# REVIEWS_FOLDER = "/media/jblamare/SAMSUNG/Pitchfork/npy/"


# data loading TVN

TVN_PREFIX = "/home/teven/Projet747/"
DATA_FOLDER = TVN_PREFIX + "Full dataset/"

MTAT_MP3_FOLDER = TVN_PREFIX + "MusicDatasets/MTAT/mp3/"
MTAT_NPY_FOLDER = TVN_PREFIX + "MusicDatasets/MTAT/npy/"
MTAT_SPLIT_FOLDER = TVN_PREFIX + "music_dataset_split/MTAT_split/"
MTAT_GENERATION_SPLIT = TVN_PREFIX + 'music_dataset_split/MTAT_generation_split'

MSD_DATABASE_FOLDER = TVN_PREFIX + "MusicDatasets/MSD/MillionSongDataset/AdditionalFiles/"
MSD_SONGS_FOLDER = TVN_PREFIX + "MusicDatasets/MSD/MillionSongDataset/"
MSD_MP3_FOLDER = TVN_PREFIX + "MusicDatasets/MSD/songs/"
MSD_NPY_FOLDER = TVN_PREFIX + "MusicDatasets/MSD/npy/"
MSD_SPLIT_FOLDER = TVN_PREFIX + "music_dataset_split/MSD_split/"

CLIP_INFO_FILE = TVN_PREFIX + "clip_info_final.csv"
REVIEWS_FOLDER = TVN_PREFIX + "MusicDatasets/MSD/Reviews/"

PITCHFORK_DB_PATH = TVN_PREFIX + "MusicDatasets/MSD/Reviews/database.sqlite"
PITCHFORK_CSV_PATH = TVN_PREFIX + "MusicDatasets/MSD/Reviews/p4kreviews.csv"

MODEL_FOLDER = TVN_PREFIX + "models/"
ENCODER_FOLDER = MODEL_FOLDER + "encoder/"
DECODER_FOLDER = MODEL_FOLDER + "decoder/"

# data loading AWS

# AWS_FOLDER = "/home/ubuntu/Projet747/"
# DATA_FOLDER = AWS_FOLDER + "Full dataset/"
# MTAT_MP3_FOLDER = AWS_FOLDER + "MusicDatasets/MTAT/mp3/"
# MTAT_NPY_FOLDER = AWS_FOLDER + "MusicDatasets/MTAT/npy/"
# MTAT_SPLIT_FOLDER = AWS_FOLDER + "music_dataset_split/MTAT_split/"
# MTAT_GENERATION_SPLIT = AWS_FOLDER + 'music_dataset_split/MTAT_generation_split'
# MSD_DATABASE_FOLDER = AWS_FOLDER + "MusicDatasets/MSD/MillionSongDataset/AdditionalFiles/"
# MSD_SONGS_FOLDER = AWS_FOLDER + "MusicDatasets/MSD/MillionSongDataset/"
# MSD_SPLIT_FOLDER = AWS_FOLDER + "music_dataset_split/MSD_split/"
# CLIP_INFO_FILE = AWS_FOLDER + "clip_info_final.csv"
# REVIEWS_FOLDER = AWS_FOLDER + "MusicDatasets/MSD/Reviews/"
# PITCHFORK_DB_PATH = AWS_FOLDER + "MusicDatasets/MSD/Reviews/database.sqlite"

# data loading AWS

# AWS_FOLDER = "/home/ubuntu/Projet747/"
# DATA_FOLDER = AWS_FOLDER + "Full dataset/"
# MTAT_MP3_FOLDER = AWS_FOLDER + "MusicDatasets/MTAT/mp3/"
# MTAT_NPY_FOLDER = AWS_FOLDER + "MusicDatasets/MTAT/npy/"
# MTAT_SPLIT_FOLDER = AWS_FOLDER + "music_dataset_split/MTAT_split/"
# MTAT_GENERATION_SPLIT = AWS_FOLDER + 'music_dataset_split/MTAT_generation_split'
# MSD_DATABASE_FOLDER = AWS_FOLDER + "MusicDatasets/MSD/MillionSongDataset/AdditionalFiles/"
# MSD_SONGS_FOLDER = AWS_FOLDER + "MusicDatasets/MSD/MillionSongDataset/"
# MSD_SPLIT_FOLDER = AWS_FOLDER + "music_dataset_split/MSD_split/"
# CLIP_INFO_FILE = AWS_FOLDER + "clip_info_final.csv"
# REVIEWS_FOLDER = AWS_FOLDER + "MusicDatasets/MSD/Reviews/"
# PITCHFORK_DB_PATH = AWS_FOLDER + "MusicDatasets/MSD/Reviews/database.sqlite"

# DATA_FOLDER = os.path.join('../mp3/')

number_sets = 1
number_labels = 50
n_songs = 2000
normalization = True

# model training

seed = 43
batch_size = 64
epochs = 10
learning_rate = 0.1  # 0.0005 (Adam)
momentum = 0.9
