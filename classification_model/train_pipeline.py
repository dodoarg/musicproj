from config.core import config

from pipeline import popularity_pipe
from processing.data_manager import load_dataset
from processing.preprocessing import binarize_popularity, balance_dataset

from sklearn.model_selection import train_test_split


def run_traning() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # preprocess target
    data[config.model_config.target_bin] = binarize_popularity(
        data[config.model_config.target_int]
    )

    data = balance_dataset(data, config.model_config.target_bin)

    data.drop([config.model_config.target_int], axis=1, inplace=True)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target_bin],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # fit model
    popularity_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=popularity_pipe)


if __name__ == "__main__":
    run_training()
