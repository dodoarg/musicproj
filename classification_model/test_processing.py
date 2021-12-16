import pandas as pd

from classification_model.processing.data_manager import *

def test_load_datasets():
    assert load_dataset(file_name="train.json")
    df = load_dataset(file_name="train.json")
    assert isinstance(df, pd.DataFrame)