from utils import *
from constants import *
import logging

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
    assert get_results(client, query, offset=25) == get_results(client, query, offset=25)


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

def test_get_song_attributes():
    random_song = get_random_song(create_client())
    assert get_song_attributes(random_song)
    song_attr = get_song_attributes(random_song)
    assert isinstance(song_attr, dict)
    LOGGER.info(song_attr)
    assert list(song_attr.keys()) == ATTRIBUTES