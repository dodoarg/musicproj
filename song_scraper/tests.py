from utils import *
from constants import *
from scrape import *

import logging
import pytest
import json

import numpy as np

LOGGER = logging.getLogger(__name__)

def test_is_query_valid():
    possible_queries = [
        "{} genre:\"{}\"".format(wildcard, genre)
        for wildcard in WILDCARDS for genre in GENRES
    ]
    assert generate_query() in possible_queries

def test_create_client():
    assert isinstance(create_client(), Spotify)

def test_get_results():
    client = create_client()
    query = generate_query()
    assert get_results(client, query)
    results = get_results(client, query)
    assert "tracks" in results.keys()
    assert "items" in results["tracks"].keys()


def test_get_validated_results():
    client = create_client()
    for _ in range(5):
        assert get_validated_results(client)


def test_get_random_song():
    assert get_random_song(create_client())
    random_song = get_random_song(create_client())
    assert isinstance(random_song, dict)
    assert all(key in random_song.keys() for key in ATTRIBUTES)
    assert all(bool(random_song[key]) for key in ATTRIBUTES)
    assert random_song["album"]["release_date"]

def test_get_musicality_features():
    client = create_client()
    random_song = get_random_song(client)
    assert get_musicality_features(client, random_song["uri"])
    random_uris = [get_random_song(client)["uri"] for _ in range(2)]
    assert get_musicality_features(client, random_uris)
    musicality_feats = get_musicality_features(client, random_song["uri"])
    assert isinstance(musicality_feats, list)
    assert isinstance(musicality_feats[0], dict)
    assert not set(musicality_feats[0].keys()) - set(MUSICALITY_FEATURES)

def test_get_song_attributes():
    random_song = get_random_song(create_client())
    assert get_song_attributes(random_song)
    song_attr = get_song_attributes(random_song)
    assert isinstance(song_attr, dict)
    assert set(song_attr.keys()) == set(ATTRIBUTES + ["year"])

def test_get_song_sample():
    random_song = get_random_song(create_client())
    wav = get_song_sample(random_song["preview_url"])
    assert isinstance(wav, io.BytesIO)

def test_load_song():
    random_song = get_random_song(create_client())
    assert load_song(random_song["preview_url"])
    y, sr = load_song(random_song["preview_url"])
    assert isinstance(y, np.ndarray)
    assert isinstance(sr, int)

def test_extract_features():
    random_song = get_random_song(create_client())
    amplitudes, sample_rate = load_song(random_song["preview_url"])
    assert extract_features(amplitudes, sample_rate)
    features_dict = extract_features(amplitudes, sample_rate)
    assert isinstance(features_dict, dict)
    assert not set(features_dict.keys()) - set(AUDIO_FEATURES)

def test_script():
    test_path = DATA_PATH / 'test.json'
    test_path.unlink(missing_ok=True)
    args = ['--n-songs', '2', '--file-name', 'test.json']
    assert parse_args(args)
    with pytest.raises(SystemExit):
        assert parse_args(args + ['--not-expected'])
    args = parse_args(args)
    main(args)
    assert test_path.exists()
    assert len(json.load(open(test_path))) == 2
    main(args)
    assert len(json.load(open(test_path))) == 4


