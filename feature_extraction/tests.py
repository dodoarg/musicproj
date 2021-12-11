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

def load_song():
    song_url = load_song_attributes(PATH_TO_EXAMPLE)[0]["preview_url"]
    assert load_song(song_url)
    y, sr = load_song(song_url)
    assert isinstance(y, np.array)
    assert isinstance(sr, int)