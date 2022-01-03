from copy import deepcopy

import pytest
from app import __version__ as _api_version
from app.config import settings
from classification_model import __version__ as model_version


def test_health(client, base_url):
    url = f"{base_url}/health"
    response = client.get(url)
    assert response.status_code == 200
    health = response.json()
    assert isinstance(health, dict)
    assert health["name"] == settings.PROJECT_NAME
    assert health["api_version"] == _api_version
    assert health["model_version"] == model_version


@pytest.mark.parametrize(
    "invalid_urls",
    [
        "http//localhost:9284/api/v1/health",
        "http//localhost:8001/api/v10/health",
        "http//localhost:8001/api/v1/not_a_valid_endpoint",
    ],
)
def test_predict_error_404(client, invalid_urls):
    response = client.post(invalid_urls)
    assert response.status_code == 404


def test_predict_raises_error_422(client, base_url, input_sample):
    payload = deepcopy(input_sample)
    payload["inputs"][0]["key"] = "not_an_int"

    url = f"{base_url}/predict"
    response = client.post(url, json=payload)
    assert response.status_code == 422


def test_predict_response_200(client, base_url, input_sample, caplog):
    payload = deepcopy(input_sample)
    url = f"{base_url}/predict"
    response = client.post(url, json=payload)
    assert "Making predictions on inputs: " in caplog.records[0].message
    assert "Prediction results" in caplog.records[1].message
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["errors"] is None
    assert prediction_data["version"] == model_version
    assert len(prediction_data["predictions"]) == 499
