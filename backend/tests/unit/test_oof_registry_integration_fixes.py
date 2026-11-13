"""Regression tests for registry fixes made during the Prototype integration.

horizon_hours: the manifest contract and register_oof_artifacts.resolve_real_run require an
integer, but validate_real_oof_context accepted only strings (Role 5's own unit tests pass "24"),
so a real manifest could never pass both checks and no OOF artifact could ever be published.
"""

from __future__ import annotations

import pytest

# tests/ has no __init__.py; pytest puts tests/unit on sys.path, so import by module name.
from test_oof_registry import _fake_run, _real_manifest

from fireguard.services.oof_registry import RegistryValidationError, validate_real_oof_context


def test_integer_horizon_hours_is_accepted_like_the_cli_requires():
    manifest = _real_manifest()
    manifest["horizon_hours"] = 24  # what register_oof_artifacts.py insists on
    validate_real_oof_context(_fake_run(), manifest)


def test_string_horizon_hours_still_accepted_for_existing_callers():
    validate_real_oof_context(_fake_run(), _real_manifest())  # "24"


def test_wrong_integer_horizon_is_still_rejected():
    manifest = _real_manifest()
    manifest["horizon_hours"] = 48
    with pytest.raises(RegistryValidationError, match="horizon_hours mismatch"):
        validate_real_oof_context(_fake_run(), manifest)


def test_boolean_horizon_is_not_treated_as_an_integer():
    manifest = _real_manifest()
    manifest["horizon_hours"] = True
    with pytest.raises(RegistryValidationError):
        validate_real_oof_context(_fake_run(), manifest)
