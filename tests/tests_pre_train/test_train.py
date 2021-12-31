from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from classification_model.training import train_pipeline_on_training_data


def try_predict(pipeline, X):
    try:
        pipeline.predict(X)
    except NotFittedError as e:
        return e


def test_training():
    fitted_pipeline, X_test, y_test = train_pipeline_on_training_data(
        _return_test_set=True
    )
    assert isinstance(fitted_pipeline, Pipeline)
    assert not isinstance(try_predict(fitted_pipeline, X_test), NotFittedError)
