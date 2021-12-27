from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str, _dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    df = pd.read_json(Path(f"{_dataset_dir}/{file_name}"))
    return df


def save_pipeline(
    *,
    pipeline_to_persist: Pipeline,
    _trained_model_dir: Path = TRAINED_MODEL_DIR,
    _save_file_name: str = f"{config.app_config.pipeline_save_file}{_version}.pkl",
) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previously
    saved models. This ensures that when the package is published
    there is only one trained model that can be called and
    we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_path = _trained_model_dir / _save_file_name

    remove_old_pipelines(
        files_to_keep=[_save_file_name], _trained_model_dir=_trained_model_dir
    )
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(
    *, file_name: str, _trained_model_dir: Path = TRAINED_MODEL_DIR
) -> Pipeline:
    """Load a persisted pipeline"""

    file_path = _trained_model_dir / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(
    *, files_to_keep: List[str], _trained_model_dir: Path = TRAINED_MODEL_DIR
) -> None:
    """Removes old model pipelines.
    This is to ensure there is a simple one-to-one mapping
    between the package version and the model version
    to be imported and used by other applications
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in _trained_model_dir.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
