import sys

from utils import (
    load_song_attributes,
    to_csv
)


def main():
    path_to_data = sys.argv[1]
    path_dest = sys.argv[2]

    songs_data = load_song_attributes(path_to_data)
    to_csv(songs_data, file_name=path_dest)


if __name__ == "__main__":
    main()