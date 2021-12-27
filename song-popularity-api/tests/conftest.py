import pytest

import logging
from _pytest.logging import caplog as _caplog
from loguru import logger

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as _client:
        yield _client


@pytest.fixture(scope="module")
def base_url():
    url = f"https://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}"
    return url


@pytest.fixture(scope="module")
def input_sample():
    return {
        "inputs": [
            {
                "acousticness": 5.0,
                "danceability": 18.2,
                "energy": 23.4,
                "instrumentalness": 76.2,
                "key": 8,
                "liveness": 9.56,
                "loudness": 0.3,
                "mode": 1,
                "speechiness": 13.3,
                "tempo": 4.5,
                "time_signature": 3,
                "valence": 55.1,
                "beats_count": 20,
                "chroma_stft_mean": 3.3,
                "root_mean_square_mean": 3.3,
                "spectral_centroid_mean": 3.3,
                "spectral_bandwidth_mean": 3.3,
                "rolloff_mean": 3.3,
                "zero_crossing_rate_mean": 3.3,
                "mfcc_1_mean": 3.3,
                "mfcc_2_mean": 3.3,
                "mfcc_3_mean": 3.3,
                "mfcc_4_mean": 3.3,
                "mfcc_5_mean": 3.3,
                "mfcc_6_mean": 3.3,
                "mfcc_7_mean": 3.3,
                "mfcc_8_mean": 3.3,
                "mfcc_9_mean": 3.3,
                "mfcc_10_mean": 3.3,
                "mfcc_11_mean": 3.3,
                "mfcc_12_mean": 3.3,
                "mfcc_13_mean": 3.3,
                "mfcc_14_mean": 3.3,
                "mfcc_15_mean": 3.3,
                "mfcc_16_mean": 3.3,
                "mfcc_17_mean": 3.3,
                "mfcc_18_mean": 3.3,
                "mfcc_19_mean": 3.3,
                "mfcc_20_mean": 3.3,
            }
        ]
    }


@pytest.fixture
def caplog(_caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            record.from_loguru = True
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message} {extra}")
    yield _caplog
    logger.remove(handler_id)
