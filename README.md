(Awkwardly) mono-repo huge project for learning purposes. 

Contains code for:
* collecting a number of pseudo-random songs from the Spotify API through [spotipy](https://github.com/plamere/spotipy). The dataset is then comprised of both musicality features fetched from the API and audio features extracted through [librosa](https://github.com/librosa/librosa) from a 30 sec long audio sample.
* the (pretty much toy) model to classify a song as popular or unpopular based on the above features. The model has been currently published to the TestPyPi index.
* the heroku-deployed app to make predictions through a web-served REST API.

Todos (?):
* Rework the app or add an endpoint to make predictions based on the author/song name (provided it is on Spotify)
* Make it work with audio features only by adding a post method to upload an audio sample (and training a "light" model along with the full one)
