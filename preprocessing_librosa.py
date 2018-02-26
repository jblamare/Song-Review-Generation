import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

n_mels = 40
n_mfcc = 128
sr = 22050
n_fft = 1024
hop_length = 512
window = "hann"
C = 10

audio_path = "../data/mp3/0/american_bach_soloists-joseph_haydn__masses-01-kyrie__allegro_moderato-146-175.mp3"
y, sr = librosa.load(audio_path, sr=sr)

D = np.abs(librosa.stft(y, window=window, n_fft=n_fft, hop_length=hop_length))**2
D = np.log(1 + C * D)
S = librosa.feature.melspectrogram(S=D, y=y, n_mels=n_mels)
feats = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
print(feats.shape)

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
