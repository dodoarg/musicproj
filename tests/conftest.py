import pytest
from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset
from classification_model.training import train_pipeline_on_training_data


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)


@pytest.fixture()
def fitted_pipeline_and_split_dataset():
    fitted_pipeline, X_test, y_test = train_pipeline_on_training_data(
        _return_test_set=True
    )
    return fitted_pipeline, X_test, y_test
