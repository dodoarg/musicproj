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
