import io

import numpy as np

from utils import *

PATH_TO_EXAMPLE = "data_example.json"

def test_load_songs():
    assert load_song_attributes(PATH_TO_EXAMPLE)
    songs = load_song_attributes(PATH_TO_EXAMPLE)
    assert isinstance(songs, list)
    assert all(isinstance(song, dict) for song in songs)
    

def test_get_song_sample():
    songs = load_song_attributes(PATH_TO_EXAMPLE)
    wav = get_song_sample(songs[0]["preview_url"])
    assert isinstance(wav, io.BytesIO)

def test_load_song():
    song_url = load_song_attributes(PATH_TO_EXAMPLE)[0]["preview_url"]
    assert load_song(song_url)
    y, sr = load_song(song_url)
    assert isinstance(y, np.ndarray)
    assert isinstance(sr, int)

def test_extract_features():
    amplitudes, sample_rate = load_song(
        load_song_attributes(PATH_TO_EXAMPLE)[0]["preview_url"]
    )
    assert extract_features(amplitudes, sample_rate)
    features_dict = extract_features(amplitudes, sample_rate)
    assert isinstance(features_dict, dict)

def test_to_csv():
    songs_data = load_song_attributes(PATH_TO_EXAMPLE)
    assert to_csv(songs_data, file_name="data_example.csv")