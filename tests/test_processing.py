import pandas as pd

from classification_model.processing.data_manager import load_dataset
from classification_model.processing.preprocessing import (
    binarize_popularity,
    balance_dataset
)

def test_load_datasets():
    df = load_dataset(file_name="train.json")
    assert isinstance(df, pd.DataFrame)

def test_binarize_popularity():
    int_series = pd.Series([1, 2, 2, 2, 2, 3, 4, 4, 7, 9, 10])
    binarized_series = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    pd.testing.assert_series_equal(
        binarize_popularity(int_series),
        binarized_series
    )

def test_balance_data():
    toy_df = pd.DataFrame({
        "var1": "s o m e t h i n g ! !".split(),
        "is_popular": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    })
    balanced_df = balance_dataset(toy_df, "is_popular")
    pd.testing.assert_series_equal(
        balanced_df.is_popular.value_counts(),
        pd.Series([3,3], index=[0,1], name="is_popular")
    )

