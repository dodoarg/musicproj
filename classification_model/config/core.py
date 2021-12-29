from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

# Project Directories
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str
    test_data_predictions: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering
    """

    target_int: str
    target_bin: str
    features: Sequence[str]
    test_size: float
    random_state: int
    categorical_features: Sequence[str]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file(cfg_path: Path = CONFIG_FILE_PATH) -> Path:
    """Locate the configuration file"""
    if cfg_path.is_file():
        return cfg_path
    raise Exception(f"Config not found at {cfg_path}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration"""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:

        if Path(cfg_path).exists():
            with open(cfg_path, "r") as cfg_file:
                parsed_config = load(cfg_file.read())

        else:
            raise OSError(f"Did not find config file at path: {cfg_path}")

    return parsed_config


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )
    return _config


config = create_and_validate_config()
