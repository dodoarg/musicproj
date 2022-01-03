import logging

import numpy as np
import pytest
from _pytest.logging import caplog as _caplog  # noqa: F401
from app.config import settings
from app.main import app
from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset
from fastapi.testclient import TestClient
from loguru import logger


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as _client:
        yield _client


@pytest.fixture(scope="module")
def base_url():
    url = f"https://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}"
    return url


@pytest.fixture()
def input_sample():
    df = load_dataset(file_name=config.app_config.test_data_file)
    input_data = {"inputs": df.replace({np.nan: None}).to_dict(orient="records")}
    return input_data


@pytest.fixture
def caplog(_caplog):  # noqa: F811
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            record.from_loguru = True
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message} {extra}")
    yield _caplog
    logger.remove(handler_id)
