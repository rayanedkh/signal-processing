import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import IPython.display as ipd

# Charger l'audio
y, sr = librosa.load("gamme_guitare_reel_do_majeur.wav")
print("Fréquence d'échantillonnage : {}, Nombres d'échantillons : {}".format(sr, y.shape[0]))

# Paramètres STFT
nfft = 1024
nperseg = 1024
noverlap = nperseg // 2
nhopsize = nperseg - noverlap

# Calculer et afficher le spectrogramme
frequencies, times, spectrogram = sig.stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max), sr=sr, hop_length=nhopsize, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# Détecter les changements de note en calculant l'énergie par tranche de temps
energy = np.sum(np.abs(spectrogram), axis=0)
energy_diff = np.diff(energy)

# Seuils pour la détection des changements de note
threshold = np.std(energy_diff) * 1.5

# Points de temps où une nouvelle note commence
note_starts = np.where(energy_diff > threshold)[0] + 1  # Ajouter 1 car np.diff réduit la dimension de 1

# Ajouter le début et la fin de l'audio pour compléter les segments
note_starts = np.concatenate(([0], note_starts, [len(times)]))

# Extraire les segments de notes
note_segments = []
for i in range(len(note_starts)-1):
    start_idx = note_starts[i] * nhopsize
    end_idx = note_starts[i+1] * nhopsize
    note_segment = y[start_idx:end_idx]
    note_segments.append(note_segment)

# Jouer et/ou sauvegarder les segments
for i, segment in enumerate(note_segments):
    print(f"Segment {i+1}:")
    ipd.display(ipd.Audio(segment, rate=sr))
    # Sauvegarder le segment si nécessaire
    sf.write(f'segment_{i+1}.wav', segment, sr)
