import numpy as np

from classification_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # one row contains a missing value
    expected_no_predictions = 499

    # very leniant
    expected_46th_prediction = 0
    expected_439th_prediction = 1

    result = make_prediction(input_data=sample_input_data)
    assert isinstance(result, dict)

    # binary predictions
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (expected_no_predictions,)
    assert isinstance(predictions[0], np.int64)
    assert predictions[46] == expected_46th_prediction
    assert predictions[439] == expected_439th_prediction

    # probabilities
    popular_proba = result.get("popular_proba")
    assert isinstance(popular_proba, np.ndarray)
    assert popular_proba.shape == (expected_no_predictions,)
    assert isinstance(popular_proba[0], np.float64)