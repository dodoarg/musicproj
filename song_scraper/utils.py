import random

from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import Spotify

from constants import WILDCARDS, GENRES, ATTRIBUTES

def generate_query():
    wildcard = random.choice(WILDCARDS)
    genre = random.choice(GENRES)
    q = "{} genre:\"{}\"".format(wildcard, genre)
    return q

def create_client():
    spotify = Spotify(client_credentials_manager=SpotifyClientCredentials())
    return spotify

def get_results(client, query, offset=None):
    results = client.search(
        query,
        limit=50,
        offset=random.randrange(1,500) if offset is None else offset,
        type='track'
    )
    return results

def get_validated_results(client):
    results = {"tracks": {"items": []}}
    while not results["tracks"]["items"]:
        query = generate_query()
        results = get_results(client, query)
    return results["tracks"]["items"]

def get_random_song(client):
    songs = get_validated_results(client)
    while len(songs) > 0:
        song_idx = random.choice(range(len(songs)))
        random_song = songs[song_idx]
        if all(random_song[attr] for attr in ATTRIBUTES):
            return random_song
        songs.remove(songs[song_idx])

def get_musicality_features(client, song_uri):
        audio_features = client.audio_features(song_uri)
        return audio_features

def get_song_attributes(song):
    attr_dict = {
        "album": song["album"]["name"],
        "artists": song["artists"][0]["name"],
        "name": song["name"],
        "popularity": song["popularity"],
        "preview_url": song["preview_url"]
    }
    return attr_dict