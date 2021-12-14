import json
import sys
from pathlib import Path

from utils import (
    create_client,
    get_random_song,
    get_song_attributes,
    get_musicality_features,
    get_audio_features
)

DATA_PATH = Path('..\data')

def main():
    spotify = create_client()
    n_songs = int(sys.argv[1])
    file_name = sys.argv[2]
    path_to_file = DATA_PATH / file_name
    if path_to_file.exists():
        songs = json.load(open(path_to_file))
    else:
        songs = []
    for n in range(n_songs):
        if (n+1) % 50 == 0:
            print(f'scraping {n+1}th song...')
        random_song = None
        while random_song is None:
            random_song = get_random_song(spotify)
        song_attributes = get_song_attributes(random_song)
        musicality_features = get_musicality_features(spotify, random_song["uri"])[0]
        audio_features = get_audio_features(random_song["preview_url"])
        songs.append({**song_attributes, **musicality_features, **audio_features})
    with open(path_to_file, "w") as file:
        json.dump(songs, file)


if __name__ == "__main__":
    main()