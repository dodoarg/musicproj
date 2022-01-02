from pathlib import Path

import pytest
from classification_model.config.core import (Config,
                                              create_and_validate_config,
                                              fetch_config_from_yaml,
                                              find_config_file)
from pydantic import ValidationError
from strictyaml import YAML

TEST_CONFIG_TEXT_BEG = """
package_name: classification_model

training_data_file: train.json
test_data_file: test.json
test_data_predictions: test_data_predictions.csv

target_int: popularity
target_bin: is_popular

pipeline_name: classification_model
pipeline_save_file: classification_model_output_v

categorical_features:
  - feature1

features:
  - feature1
"""

MISSING_ENTRY_TEST_CONFIG_TEXT = f"""
{TEST_CONFIG_TEXT_BEG}

test_size: 0.1
"""

INVALID_ENTRY_TEST_CONFIG_TEXT = f"""
{TEST_CONFIG_TEXT_BEG}

test_size: 0.05

random_state: 0
"""


MISTYPED_ENTRY_TEST_CONFIG_TEXT = f"""
{TEST_CONFIG_TEXT_BEG}

test_size: not_a_float

random_state: 0
"""

VALID_TEST_CONFIG_TEXT = f"""
{TEST_CONFIG_TEXT_BEG}

test_size: 0.1

random_state: 0
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


def _get_parsed_config(config_path, config_text):
    config_path.write_text(config_text)
    parsed_config = fetch_config_from_yaml(config_path)
    return parsed_config


@pytest.mark.parametrize(
    "invalid_config_text, error_strings",
    [
        (MISSING_ENTRY_TEST_CONFIG_TEXT, ["field required", "random_state"]),
        (INVALID_ENTRY_TEST_CONFIG_TEXT, ["test_size must be at least 0.1"]),
        (MISTYPED_ENTRY_TEST_CONFIG_TEXT, ["test_size", "not a valid float"]),
    ],
)
def test_create_and_validate_config_raises_validation_error(
    tmp_config_path, invalid_config_text, error_strings
):
    parsed_config = _get_parsed_config(tmp_config_path, invalid_config_text)
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)
    error_msg = str(excinfo.value)
    assert all(err in error_msg for err in error_strings)


def test_create_and_validate_config(tmp_config_path):
    parsed_config = _get_parsed_config(tmp_config_path, VALID_TEST_CONFIG_TEXT)
    config = create_and_validate_config(parsed_config=parsed_config)
    assert isinstance(config, Config)
    assert config.app_config
    assert config.model_config
