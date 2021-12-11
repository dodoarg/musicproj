import json
import io
import librosa
import pydub

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
    y, sr = librosa.load(get_song_sample(song_url))