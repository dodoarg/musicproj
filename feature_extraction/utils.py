import json
import io
import librosa
import pydub
import csv

import numpy as np

from urllib.request import urlopen

def load_song_attributes(path):
    with open(path) as file:
        songs = json.load(file)
    return songs

def get_song_sample(song_url):
    wav = io.BytesIO()
    with urlopen(song_url) as r:
        r.seek = lambda *args: None # allows pydub to call seek(0)
        pydub.AudioSegment.from_file(r).export(wav, "wav")
    return wav

def load_song(song_url):
    y, sr = librosa.load(
        get_song_sample(song_url),
        mono=True,
        duration=30)
    return y, sr

def extract_features(amplitudes, sample_rate):
    tempo, beats = librosa.beat.beat_track(y=amplitudes, sr=sample_rate)
    chroma_stft = librosa.feature.chroma_stft(y=amplitudes, sr=sample_rate)
    rms = librosa.feature.rms(y=amplitudes)
    spec_cent = librosa.feature.spectral_centroid(y=amplitudes, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=amplitudes, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=amplitudes, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(amplitudes)
    mfcc = librosa.feature.mfcc(y=amplitudes, sr=sample_rate)
    features_dict = {
        "tempo": tempo,
        "beats": beats.shape[0],
        "chroma_stft": np.mean(chroma_stft),
        "root-mean-square": np.mean(rms),
        "spectral_centroid": np.mean(spec_cent),
        "spectral_bandwidth": np.mean(spec_bw),
        "rolloff": np.mean(rolloff),
        "zero_crossing_rate": np.mean(zcr),
    }
    mfccs = {f'mfcc_{i}' : np.mean(coef) for i,coef in enumerate(mfcc)}
    return {**features_dict, **mfccs}

def to_csv(songs_data, file_name="data.csv"):
    header = 'album artists name popularity tempo beats chroma_stft root-mean-square \
              spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1,21):
        header += f' mfcc_{i}'
    header = header.split()
    with open(file_name, "w", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i_song,song_data in enumerate(songs_data):
            line = [song_data[name] for name in header[:4]]
            y, sr = load_song(song_data["preview_url"])
            features = extract_features(y, sr)
            line += list(features.values())
            writer.writerow(line)
            if (i_song+1) % 50 == 0:
                print(f'doing song {i_song+1}...')
    return 1
