from pathlib import Path
import pandas as pd

from classification_model.config.core import DATASET_DIR

def load_dataset(*, file_name: str) -> pd.DataFrame:
    df = pd.read_json(Path(f'{DATASET_DIR}/{file_name}'))
    return df