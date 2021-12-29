import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error as mse

from classification_model.config.core import PACKAGE_ROOT, config
from classification_model.predict import make_prediction
from classification_model.processing.data_manager import load_dataset


@pytest.mark.differential
def test_model_prediction_differential():
    """
    This test compares predictions across two
    adjacent releases of the model
    """

    previous_model_df = pd.read_csv(
        f"{PACKAGE_ROOT}/{config.app_config.test_data_predictions}"
    )
    previous_model_proba = previous_model_df["popular_proba"].values

    test_data = load_dataset(file_name=config.app_config.test_data_file)
    multiple_test_input = test_data[85:390]
    current_result = make_prediction(input_data=multiple_test_input)
    current_model_proba = current_result.get("popular_proba")

    assert len(previous_model_proba) == len(current_model_proba)
    # very leniant. Should maybe look at the real values for the test set?
    assert mse(previous_model_proba, current_model_proba) < 0.3
