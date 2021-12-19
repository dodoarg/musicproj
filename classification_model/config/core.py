from pathlib import Path
from typing import List, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

# Project Directories
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"

class ModelConfig(BaseModel):
    target: str
    features: List[str]
    test_size: float
    random_state: int
    categorical_vars: Sequence[str]


def find_config_file(
    cfg_path=CONFIG_FILE_PATH
) -> Path:
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


def create_and_validate_config(
    parsed_config: YAML = None
) -> ModelConfig:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

config = create_and_validate_config()