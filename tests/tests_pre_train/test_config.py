from pathlib import Path

import pytest
from classification_model.config.core import (CONFIG_FILE_PATH, PACKAGE_ROOT,
                                              Config,
                                              create_and_validate_config,
                                              fetch_config_from_yaml,
                                              find_config_file)
from strictyaml import YAML


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


def test_create_and_validate_config():
    assert isinstance(create_and_validate_config(), Config)
