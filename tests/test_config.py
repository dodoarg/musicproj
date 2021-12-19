import pytest

from strictyaml import YAML

from classification_model.config.core import (
    PACKAGE_ROOT,
    CONFIG_FILE_PATH,
    find_config_file,
    fetch_config_from_yaml,
    create_and_validate_config,
    Config
)

def test_find_config_file():
    fake_path = PACKAGE_ROOT / "not_a_config_file"
    with pytest.raises(Exception):
        find_config_file(cfg_path=fake_path)
    real_path = CONFIG_FILE_PATH
    assert find_config_file() == real_path

def test_fetch_config_from_yaml():
    fake_path = PACKAGE_ROOT / "does not exist"
    with pytest.raises(OSError):
        fetch_config_from_yaml(cfg_path=fake_path)
    assert isinstance(fetch_config_from_yaml(), YAML)

def test_create_and_validate_config():
    assert isinstance(create_and_validate_config(), Config)
