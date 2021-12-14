import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from utils import (
    create_client,
    get_random_song,
    get_song_attributes,
    get_musicality_features,
    get_audio_features
)

DATA_PATH = Path('..\data')

def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('--n-songs', type=int, help='number of songs to be scraped')
    parser.add_argument('--file-name', type=str, help='name of json file to contain the data')
    return parser.parse_args(args)

def main(args):
    spotify = create_client()
    #n_songs = int(sys.argv[1])
    #file_name = sys.argv[2]
    path_to_file = DATA_PATH / args.file_name
    if path_to_file.exists():
        songs = json.load(open(path_to_file))
    else:
        songs = []
    for n in range(args.n_songs):
        if (n+1) % 50 == 0:
            print(f'scraping {n+1}th song...')
        random_song = get_random_song(spotify)
        song_attributes = get_song_attributes(random_song)
        musicality_features = get_musicality_features(spotify, random_song["uri"])[0]
        audio_features = get_audio_features(random_song["preview_url"])
        songs.append({**song_attributes, **musicality_features, **audio_features})
    with open(path_to_file, "w") as file:
        json.dump(songs, file)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)