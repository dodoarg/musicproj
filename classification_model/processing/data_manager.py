from pathlib import Path
import pandas as pd

from classification_model.config.core import DATASET_DIR

def load_dataset(
    *,
    file_name: str,
    dataset_dir: Path = DATASET_DIR
) -> pd.DataFrame:
    df = pd.read_json(Path(f'{dataset_dir}/{file_name}'))
    return df