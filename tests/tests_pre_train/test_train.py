import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
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

    # benchmarking against random baseline
    y_predicted = fitted_pipeline.predict(X_test)
    baseline = np.random.randint(0, 2, len(y_test))

    baseline_acc = accuracy_score(y_test, baseline)
    model_acc = accuracy_score(y_test, y_predicted)

    assert model_acc >= baseline_acc
