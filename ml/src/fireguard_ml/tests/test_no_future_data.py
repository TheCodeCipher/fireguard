"""No-look-ahead guards: the strict channel list can only contain Level-A PASS channels, never GFS,
and the other roles' configs agree with the register."""

from __future__ import annotations

import pytest

from fireguard_ml.data.channel_policy import (
    assert_strict_channels,
    load_register,
    strict_channels,
)

# Decided 2026-11-13 from the evidence in docs/leakage_audit.csv. If a channel's status changes,
# update this list in the same commit as the register so the change is reviewed deliberately.
EXPECTED_STRICT = [
    "viirs_m11",
    "viirs_i2",
    "viirs_i1",
    "slope",
    "aspect",
    "elevation",
    "binary_active_fire_mask",
]


@pytest.fixture(scope="module")
def register(repo_root):
    return load_register(repo_root / "docs" / "leakage_audit.csv")


def test_strict_list_is_exactly_the_reviewed_list(register):
    assert strict_channels(register) == EXPECTED_STRICT


def test_every_strict_channel_passes_level_a(register):
    for name in strict_channels(register):
        assert register[name].no_lookahead == "PASS", name


def test_unknown_or_failed_level_a_is_never_strict(register):
    for r in register.values():
        if r.no_lookahead in ("UNKNOWN", "FAIL"):
            assert r.input_decision != "STRICT", r.name


def test_no_forecast_channel_is_strict(register):
    assert not [n for n in strict_channels(register) if "gfs" in n or "forecast" in n]


def test_requesting_an_excluded_channel_raises(register):
    assert_strict_channels(["binary_active_fire_mask", "elevation"], register)
    for bad in ("gfs_wind_speed", "ndvi", "land_cover_class", "wind_speed", "made_up_channel"):
        with pytest.raises(ValueError, match="not allowed"):
            assert_strict_channels(["elevation", bad], register)


def test_role3_temporal_config_excludes_exactly_the_registers_gfs_indices(register, repo_root):
    yaml = pytest.importorskip("yaml")
    cfg_path = repo_root / "configs" / "temporal_unet.yaml"
    if not cfg_path.exists():
        pytest.skip("Role 3 config not present")
    cfg = yaml.safe_load(cfg_path.read_text())
    gfs_indices = sorted(
        int(r.processed_index) for n, r in register.items() if n.startswith("gfs_")
    )
    assert sorted(cfg["data"]["excluded_processed_indices"]) == gfs_indices


def test_role3_checkpoints_refuse_forecast_channel_names():
    from fireguard_ml.models.checkpointing import FORBIDDEN_STRICT_SUBSTRINGS

    assert "gfs" in FORBIDDEN_STRICT_SUBSTRINGS
