import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

n_mels = 40
n_mfcc = 13
sr = 22050
n_fft = 1024
hop_length = 512
window = "hann"

audio_path = "../data/mp3/0/american_bach_soloists-joseph_haydn__masses-01-kyrie__allegro_moderato-146-175.mp3"
y, sr = librosa.load(audio_path, sr=22050)

D = np.abs(librosa.stft(y, window=window, n_fft=n_fft, hop_length=hop_length))**2
S = librosa.feature.melspectrogram(S=D, y=y, n_mels=n_mels)
feats = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
print(feats.shape)

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
