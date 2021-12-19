from sklearn.pipeline import Pipeline

from classification_model.pipeline import popularity_pipe

def test_pipeline():
    assert isinstance(popularity_pipe, Pipeline)