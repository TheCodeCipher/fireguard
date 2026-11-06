
from fastapi.testclient import TestClient

from fireguard.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to the FireGuard API"


def test_health_endpoint():
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["application"] == "FireGuard API"
    assert data["environment"] == "development"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data
