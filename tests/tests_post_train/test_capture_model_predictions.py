from pathlib import Path

import pandas as pd

from classification_model.config.core import ROOT
from tests.capture_model_predictions import capture_predictions

TESTS_PATH = ROOT / "tests" / "tests_post_train"


def test_capture_predictions():
    file_name = "test_test_data_predictions.csv"
    save_path = TESTS_PATH / "test_files"
    capture_predictions(_save_file=file_name, _save_path=save_path)

    assert (save_path / file_name).exists()
    predictions_df = pd.read_csv(save_path / file_name)
    assert len(predictions_df) == 305
    assert "popular_proba" in predictions_df.columns.values
    Path.unlink(save_path / file_name)
