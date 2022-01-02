import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def try_predict(pipeline, X):
    try:
        pipeline.predict(X)
    except NotFittedError as e:
        return e


def test_training(fitted_pipeline_and_split_dataset):
    fitted_pipeline, X_test, y_test = fitted_pipeline_and_split_dataset
    assert isinstance(fitted_pipeline, Pipeline)
    assert not isinstance(try_predict(fitted_pipeline, X_test), NotFittedError)

    # benchmarking against random baseline
    y_predicted = fitted_pipeline.predict(X_test)
    baseline = np.random.randint(0, 2, len(y_test))

    baseline_acc = accuracy_score(y_test, baseline)
    model_acc = accuracy_score(y_test, y_predicted)

    assert model_acc >= baseline_acc
