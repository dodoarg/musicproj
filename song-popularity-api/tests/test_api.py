from copy import deepcopy

from classification_model import __version__ as model_version

from app import __version__ as _api_version
from app.config import settings


def test_health(client):
    response = client.get("http://localhost:8001/api/v1/health")
    assert response.status_code == 200
    health = response.json()
    assert isinstance(health, dict)
    assert health["name"] == settings.PROJECT_NAME
    assert health["api_version"] == _api_version
    assert health["model_version"] == model_version


def test_predict_raises_error_422(client, input_sample):
    payload = deepcopy(input_sample)
    payload["inputs"][0]["key"] = "not_an_int"

    response = client.post("http://localhost:8001/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_response_200(client, input_sample):
    payload = deepcopy(input_sample)
    response = client.post("http://localhost:8001/api/v1/predict", json=payload)
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["errors"] is None
    assert prediction_data["version"] == model_version
    assert prediction_data["predictions"] == [1]
