"""Shared pytest fixtures for the fireguard_ml test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


def find_repo_root() -> Path:
    """Walk up from this file to the repository root (the folder holding ``ml/`` and ``docs/``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "ml").is_dir() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError("repository root not found above " + str(Path(__file__)))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return find_repo_root()
