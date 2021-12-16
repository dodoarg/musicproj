import pandas as pd

from classification_model.processing.data_manager import load_dataset

def test_load_datasets():
    df = load_dataset(file_name="train.json")
    assert isinstance(df, pd.DataFrame)