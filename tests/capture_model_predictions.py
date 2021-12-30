"""This script should only be run in CI"""


import sys
from pathlib import Path

import pandas as pd

from classification_model.config.core import PACKAGE_ROOT, config
from classification_model.predict import make_prediction
from classification_model.processing.data_manager import load_dataset


def capture_predictions(
    _file_name: str = config.app_config.test_data_file,
    _save_file: str = config.app_config.test_data_predictions,
    _save_path: Path = PACKAGE_ROOT,
) -> None:
    """Save test data predictions to a csv"""

    test_data = load_dataset(file_name=_file_name)

    multiple_test_input = test_data[85:390]

    predictions_df = pd.DataFrame(make_prediction(input_data=multiple_test_input))
    predictions_df.to_csv(f"{_save_path}/{_save_file}")


if __name__ == "__main__":
    capture_predictions(_save_path=sys.argv[1])
