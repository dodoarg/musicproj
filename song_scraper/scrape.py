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
    path_dest = sys.argv[2]
    songs = []
    for n in range(n_songs):
        if (n+1) % 50 == 0:
            print(f'scraping {n+1}th song...')
        random_song = None
        while random_song is None:
            random_song = get_random_song(spotify)
        song_attributes = get_song_attributes(random_song)
        songs.append(song_attributes)
    with open(path_dest, "w") as file:
        json.dump(songs, file)


if __name__ == "__main__":
    main()