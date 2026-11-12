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
    assert data["version"] == "0.2.0"
    assert data["dependencies"]["database"] == "healthy"


def test_health_does_not_expose_firms_key(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "test-secret-never-return-this")
    from fireguard.core.config import get_settings

    get_settings.cache_clear()
    try:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "test-secret-never-return-this" not in response.text
    finally:
        get_settings.cache_clear()
