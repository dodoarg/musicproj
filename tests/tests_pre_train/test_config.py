from pathlib import Path

import pytest
from classification_model.config.core import (Config,
                                              create_and_validate_config,
                                              fetch_config_from_yaml,
                                              find_config_file)
from pydantic import ValidationError
from strictyaml import YAML

MISSING_ENTRY_TEST_CONFIG_TEXT = """
package_name: classification_model

training_data_file: train.json
test_data_file: test.json
test_data_predictions: test_data_predictions.csv

target_int: popularity
target_bin: is_popular

pipeline_name: classification_model
pipeline_save_file: classification_model_output_v

features:
  - feature1

test_size: 0.1

categorical_features:
  - feature1
"""

INVALID_ENTRY_TEST_CONFIG_TEXT = """
package_name: classification_model

training_data_file: train.json
test_data_file: test.json
test_data_predictions: test_data_predictions.csv

target_int: popularity
target_bin: is_popular

pipeline_name: classification_model
pipeline_save_file: classification_model_output_v

features:
  - feature1

test_size: 0.05

random_state: 0

categorical_features:
  - feature1
"""


MISTYPED_ENTRY_TEST_CONFIG_TEXT = """
package_name: classification_model

training_data_file: train.json
test_data_file: test.json
test_data_predictions: test_data_predictions.csv

target_int: popularity
target_bin: is_popular

pipeline_name: classification_model
pipeline_save_file: classification_model_output_v

features:
  - feature1

test_size: not_a_float

random_state: 0

categorical_features:
  - feature1
"""


def test_find_config_file(tmpdir):
    config_dir = Path(tmpdir)
    with pytest.raises(Exception):
        find_config_file(cfg_path=config_dir)

    config_path = config_dir / "test_file.txt"
    config_path.write_text("this is a test")
    config_path_retrieved = find_config_file(cfg_path=config_path)
    assert config_path_retrieved == config_path


def test_fetch_config_from_yaml(tmpdir):
    config_dir = Path(tmpdir)
    config_path = config_dir / "sample_config.txt"
    with pytest.raises(OSError):
        fetch_config_from_yaml(cfg_path=config_path)

    config_path.write_text("this is a test")
    parsed_config = fetch_config_from_yaml(cfg_path=config_path)
    assert isinstance(parsed_config, YAML)


def test_create_and_validate_config(tmpdir):
    config_dir = Path(tmpdir)
    config_path = config_dir / "sample_config.yml"

    config_path.write_text(MISSING_ENTRY_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(config_path)
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)
    error_msg = str(excinfo.value)
    assert "field required" in error_msg
    assert "random_state" in error_msg

    config_path.write_text(INVALID_ENTRY_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(config_path)
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)
    error_msg = str(excinfo.value)
    assert "test_size must be at least 0.1" in error_msg

    config_path.write_text(MISTYPED_ENTRY_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(config_path)
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)
    error_msg = str(excinfo.value)
    assert "test_size" in error_msg
    assert "not a valid float" in error_msg

    assert isinstance(create_and_validate_config(), Config)
