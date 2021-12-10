import json
import sys

from utils import (
    create_client,
    get_random_song,
    get_song_attributes
)


def main():
    spotify = create_client()
    n_songs = int(sys.argv[1])
    songs = []
    for _ in range(n_songs):
        random_song = get_random_song(spotify)
        song_attributes = get_song_attributes(random_song)
        songs.append(song_attributes)
    with open("data.json", "w") as file:
        json.dump(songs, file)


if __name__ == "__main__":
    main()