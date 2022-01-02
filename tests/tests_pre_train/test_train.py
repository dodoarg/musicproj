import numpy as np
from pandas.api.types import is_object_dtype as is_object
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from classification_model.config.core import config


def try_predict(pipeline, X):
    try:
        pipeline.predict(X)
    except NotFittedError as e:
        return e


def test_pre_training(fitted_pipeline_and_split_dataset):
    model_config = config.model_config
    _, X_test, y_test = fitted_pipeline_and_split_dataset

    assert not any(
        target in X_test.columns
        for target in (model_config.target_int, model_config.target_bin)
    )
    assert all(
        is_object(X_test[categorical])
        for categorical in model_config.categorical_features
    )
    assert not y_test[y_test != 0][y_test != 1].count()


def test_benchmark(fitted_pipeline_and_split_dataset):
    fitted_pipeline, X_test, y_test = fitted_pipeline_and_split_dataset

    assert isinstance(fitted_pipeline, Pipeline)
    assert not isinstance(try_predict(fitted_pipeline, X_test), NotFittedError)

    # benchmarking against random baseline
    y_predicted = fitted_pipeline.predict(X_test)
    baseline = np.random.randint(0, 2, len(y_test))

    baseline_acc = accuracy_score(y_test, baseline)
    model_acc = accuracy_score(y_test, y_predicted)

    assert model_acc >= baseline_acc
