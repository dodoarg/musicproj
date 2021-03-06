import io
import random
from typing import List
from urllib.request import urlopen

import librosa
import numpy as np
import pydub
from constants import (
    WILDCARDS,
    GENRES,
    ATTRIBUTES,
    MUSICALITY_FEATURES
)
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials


def generate_query():
    wildcard = random.choice(WILDCARDS)
    genre = random.choice(GENRES)
    q = '{} genre:"{}"'.format(wildcard, genre)
    return q


def create_client():
    spotify = Spotify(client_credentials_manager=SpotifyClientCredentials())
    return spotify


def search_query(client, query, offset=None):
    results = client.search(
        query,
        limit=50,
        offset=random.randrange(1, 500) if offset is None else offset,
        type="track",
    )
    return results

def get_nonempty_items(client):
    while True:
        query = generate_query()
        results = search_query(client, query)
        items = results["tracks"]["items"]
        if items:
            return items


def get_validated_random_song_from_items(items):
    while len(items) > 0:
        song_idx = random.choice(range(len(items)))
        random_song = items[song_idx]
        if all(random_song[attr] for attr in ATTRIBUTES):
            if random_song["album"]["release_date"]:
                return random_song
        items.remove(items[song_idx])


def get_random_song(client):
    random_song = None
    while random_song is None:
        items = get_nonempty_items(client)
        random_song = get_validated_random_song_from_items(items)
    return random_song


def get_musicality_features(client, song_uri):
    audio_features = client.audio_features(song_uri)
    return [
        {
            feature: song[feature]
            for feature in MUSICALITY_FEATURES
        }
        for song in audio_features
    ]


def get_song_attributes(song):
    attr_dict = {
        "album": song["album"]["name"],
        "artists": song["artists"][0]["name"],
        "name": song["name"],
        "year": song["album"]["release_date"][:4],
        "popularity": song["popularity"],
        "preview_url": song["preview_url"],
    }
    return attr_dict


def get_song_sample(song_url):
    wav = io.BytesIO()
    with urlopen(song_url) as r:
        r.seek = lambda *args: None  # allows pydub to call seek(0)
        pydub.AudioSegment.from_file(r).export(wav, "wav")
    return wav


def load_song(song_url):
    y, sr = librosa.load(get_song_sample(song_url), mono=True, duration=30)
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
        "tempo": float(tempo),
        "beats_count": float(beats.shape[0]),
        "chroma_stft_mean": float(np.mean(chroma_stft)),
        "root_mean_square_mean": float(np.mean(rms)),
        "spectral_centroid_mean": float(np.mean(spec_cent)),
        "spectral_bandwidth_mean": float(np.mean(spec_bw)),
        "rolloff_mean": float(np.mean(rolloff)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
    }
    mfccs = {
        f"mfcc_{i+1}_mean": float(np.mean(coef))
        for i, coef in enumerate(mfcc)
    }
    return {**features_dict, **mfccs}


def get_audio_features(song_url):
    y, sr = load_song(song_url)
    audio_features_dict = extract_features(y, sr)
    return audio_features_dict
