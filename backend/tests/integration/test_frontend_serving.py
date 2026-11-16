"""FIREGUARD_FRONTEND_DIST: the API also serves the built website (scripts/app/start_app.py)."""

import importlib

from fastapi.testclient import TestClient

import fireguard.main
from fireguard.core.config import get_settings


def _app_with(monkeypatch, dist):
    if dist is None:
        monkeypatch.delenv("FIREGUARD_FRONTEND_DIST", raising=False)
    else:
        monkeypatch.setenv("FIREGUARD_FRONTEND_DIST", str(dist))
    get_settings.cache_clear()
    return importlib.reload(fireguard.main).app


def test_site_served_next_to_the_api(monkeypatch, tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>FireGuard</title>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    try:
        client = TestClient(_app_with(monkeypatch, tmp_path))
        assert "<title>FireGuard</title>" in client.get("/").text
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/api/v1/health").json()["status"] == "healthy"
        assert client.get("/docs").status_code == 200
    finally:
        _app_with(monkeypatch, None)


def test_missing_build_falls_back_to_the_api_root(monkeypatch, tmp_path):
    try:
        client = TestClient(_app_with(monkeypatch, tmp_path / "nothing"))
        assert client.get("/").json()["message"] == "Welcome to the FireGuard API"
    finally:
        _app_with(monkeypatch, None)
