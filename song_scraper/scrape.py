import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json, random

def main():
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    results = spotify.search(
        "%a% genre:\"rock\"22",
        limit=12,
        offset=0,
        type='track'
    )
    song_info = random.choice(results['tracks']['items'])
    artist = song_info['artists'][0]['name']
    song = song_info['name']
    popularity = song_info['popularity']
    print(artist, '\n', song, '\npopularity: ', popularity)
    print(song_info["preview_url"])

if __name__ == "__main__":
    main()