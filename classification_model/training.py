from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classification_model.config.core import config
from classification_model.pipeline import popularity_pipe
from classification_model.processing.data_manager import load_dataset
from classification_model.processing.preprocessing import (
    balance_dataset,
    binarize_popularity,
)


def train_pipeline_on_training_data() -> Pipeline:
    """Train the model, return fitted pipeline"""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # preprocess target
    data[config.model_config.target_bin] = binarize_popularity(
        data[config.model_config.target_int]
    )

    data = balance_dataset(data, config.model_config.target_bin)

    data.drop([config.model_config.target_int], axis=1, inplace=True)

    # cast categorical vars as object (maybe should do this in the data_manager?)
    data[config.model_config.categorical_features] = data[
        config.model_config.categorical_features
    ].astype("O")

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target_bin],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # fit model
    popularity_pipe.fit(X_train, y_train)

    return popularity_pipe
