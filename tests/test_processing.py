from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model.config.core import ROOT
from classification_model.processing.data_manager import (load_dataset,
                                                          remove_old_pipelines,
                                                          save_pipeline)
from classification_model.processing.preprocessing import (balance_dataset,
                                                           binarize_popularity)

TESTS_PATH = ROOT / "tests"


def test_load_dataset():
    file_name = "test_ds_to_load.json"
    dataset_dir = (TESTS_PATH / "test_files")
    print(Path(f"{dataset_dir}/{file_name}"))
    df = load_dataset(file_name=file_name, _dataset_dir=dataset_dir)
    assert isinstance(df, pd.DataFrame)


def test_remove_old_pipelines():
    trained_model_dir = (TESTS_PATH / "test_files" / "test_train_files")
    open(f"{trained_model_dir}/to_keep.txt", "w")
    for i in range(3):
        open(f"{trained_model_dir}/to_delete_{i}", "w")
    files_to_keep = ["to_keep.txt"]
    remove_old_pipelines(
        files_to_keep=files_to_keep, _trained_model_dir=trained_model_dir
    )
    assert (trained_model_dir / files_to_keep[0]).exists()
    assert not list(trained_model_dir.glob("to_delete*"))


def test_save_pipeline():
    trained_model_dir = (TESTS_PATH / "test_files" / "test_train_files")
    pipeline_to_persist = Pipeline([("dum", "passthrough")])
    save_file_name = "test_persisted_pipeline.pkl"
    save_pipeline(
        pipeline_to_persist=pipeline_to_persist,
        _trained_model_dir=trained_model_dir,
        _save_file_name=save_file_name,
    )
    assert (trained_model_dir / save_file_name).exists()
    (trained_model_dir / save_file_name).unlink()


def test_binarize_popularity():
    int_series = pd.Series(
        [1, 2, 2, 2, 2, 3, 4, 4, 7, 9, 10]
    )
    binarized_series = pd.Series(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    )
    pd.testing.assert_series_equal(
        binarize_popularity(int_series),
        binarized_series
    )


def test_balance_data():
    toy_df = pd.DataFrame(
        {
            "var1": "s o m e t h i n g ! !".split(),
            "is_popular": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    balanced_df = balance_dataset(toy_df, "is_popular")
    pd.testing.assert_series_equal(
        balanced_df.is_popular.value_counts(),
        pd.Series([3, 3], index=[0, 1], name="is_popular"),
    )
