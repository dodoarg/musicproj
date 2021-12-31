from sklearn.pipeline import Pipeline

from classification_model.training import train_pipeline_on_training_data


def test_training():
    fitted_pipeline = train_pipeline_on_training_data()
    assert isinstance(fitted_pipeline, Pipeline)
