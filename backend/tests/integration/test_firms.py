from datetime import date

from fastapi.testclient import TestClient

from fireguard.main import app

client = TestClient(app)
ENDPOINT = "/api/v1/firms/observations"


def test_get_all_fire_observations():
    response = client.get(ENDPOINT)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "mock"
    assert data["count"] == 4
    assert len(data["observations"]) == 4


def test_filter_high_confidence():
    response = client.get(ENDPOINT, params={"confidence": "high"})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert all(item["confidence"] == "high" for item in data["observations"])


def test_limit_observations():
    response = client.get(ENDPOINT, params={"limit": 2})
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_invalid_confidence_rejected():
    response = client.get(ENDPOINT, params={"confidence": "extreme"})
    assert response.status_code == 422


def test_invalid_limit_rejected():
    response = client.get(ENDPOINT, params={"limit": 0})
    assert response.status_code == 422


def test_observation_coordinates_are_valid():
    response = client.get(ENDPOINT)
    assert response.status_code == 200
    for item in response.json()["observations"]:
        assert -90 <= item["latitude"] <= 90
        assert -180 <= item["longitude"] <= 180
