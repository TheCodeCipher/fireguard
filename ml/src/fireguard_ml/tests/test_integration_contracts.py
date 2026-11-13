"""Cross-role contract tests for the integrated Prototype branch (added during integration).

Each role's own tests check its module against its own documents. These tests check the places
where two or more roles must agree, so a later edit by one role that silently breaks another fails
CI instead of failing at demo time. Each test names the roles involved.

Nothing here reads WSTS data; everything is from committed configs, manifests and code.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

from fireguard_ml.data.channel_policy import load_register, strict_channels
from fireguard_ml.data.channel_schema import PROCESSED_WSTS_CHANNELS
from fireguard_ml.data.padding import pad_to_multiple, pad_valid_mask
from fireguard_ml.data.wsts_dataset import DATA_CONTRACT_VERSION, PROCESSED_CHANNELS
from fireguard_ml.training.temporal_data import TemporalSample, pad_samples

# ml/src/fireguard_ml/tests/<this file> -> repository root is four levels up.
ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "configs"
MANIFESTS = ROOT / "data" / "manifests"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256_lf(path: Path) -> str:
    """Same hashing rule as Role 4's p1_split_hashes.txt: CRLF normalised to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# --------------------------------------------------------------------------- channel names


def test_role2_channel_schema_is_role1_loader_schema():
    """Roles 1+2: the U-Net adapter selects by name from the same 40 names the loader produces.

    Before integration Role 2 carried its own copy in which 24 of the 40 names differed.
    """
    assert tuple(PROCESSED_WSTS_CHANNELS) == tuple(PROCESSED_CHANNELS)
    assert len(PROCESSED_CHANNELS) == 40


def test_strict_channel_list_is_identical_in_every_role():
    """Roles 1-4: one strict list, in loader order, everywhere it is written down."""
    register = strict_channels(load_register(ROOT / "docs" / "leakage_audit.csv"))  # Role 4
    role1 = [c["name"] for c in _yaml(CONFIGS / "data" / "strict_input_v1.yaml")["channels"]]
    role2 = _yaml(CONFIGS / "unet.yaml")["channel_schema"]["selected_names"]
    role3 = _yaml(CONFIGS / "temporal_unet.yaml")["data"]["channels"]
    # The register is unordered (a set of decisions); the configs fix the loader order.
    assert sorted(register) == sorted(role1)
    assert role1 == role2 == role3
    # Every strict name must exist in the processed schema (no typos).
    assert all(name in PROCESSED_CHANNELS for name in role1)


def test_strict_list_excludes_every_gated_source():
    """Roles 3+4: GFS, GRIDMET, PDSI, NDVI/EVI2, land cover and fire time stay out."""
    strict = _yaml(CONFIGS / "temporal_unet.yaml")["data"]["channels"]
    banned = ("gfs", "precip", "wind", "temperature", "humidity", "energy_release",
              "palmer", "ndvi", "evi2", "land_cover", "active_fire_time")  # fmt: skip
    assert not [c for c in strict if any(b in c for b in banned)]


def test_data_contract_version_matches_role1_config():
    """Roles 1+3+5: the version string every export and registry record carries."""
    assert _yaml(CONFIGS / "data" / "strict_input_v1.yaml")["version"] == DATA_CONTRACT_VERSION


# ------------------------------------------------------------------------------ split hash


def test_every_config_cites_the_current_frozen_split():
    """Roles 2+4: the U-Net config hash equals the recorded and the actual manifest hash."""
    hashes = (MANIFESTS / "p1_split_hashes.txt").read_text(encoding="utf-8")
    assert "status=FROZEN" in hashes
    rows = [line.split() for line in hashes.splitlines() if not line.startswith("#")]
    recorded = next(h for h, name in (r for r in rows if r) if name == "p1_loyo_fold_manifest.csv")
    actual = _sha256_lf(MANIFESTS / "p1_loyo_fold_manifest.csv")
    assert recorded == actual
    assert _yaml(CONFIGS / "unet.yaml")["data"]["split_manifest_sha256"] == actual
    # Role 3's config does not duplicate the hash; it points at the hash file it re-verifies.
    t = _yaml(CONFIGS / "temporal_unet.yaml")["data"]
    assert t["split_hashes"] == "data/manifests/p1_split_hashes.txt"


def test_withdrawn_first_freeze_hash_is_not_used_in_configs():
    """Role 4 withdrew the first frozen hash (55a3ca18...); no config value may still cite it.

    Only YAML values are checked (comments may mention the old hash to explain the change).
    """

    def values(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from values(v)
        elif isinstance(node, list):
            for v in node:
                yield from values(v)
        else:
            yield str(node)

    for path in CONFIGS.rglob("*.yaml"):
        assert not [v for v in values(_yaml(path)) if "55a3ca18" in v], path


# --------------------------------------------------------------------------------- padding


def test_role2_and_role3_padding_are_the_same_operation():
    """Roles 2+3: M1 and M3/M4 pad bottom/right to 32 and mark the padding invalid identically.

    Role 3's pad_samples works on whole TemporalSamples; Role 2's helpers on tensors. Same frame,
    same result, so M1 and M3/M4 are scored on identical pixels.
    """
    rng = np.random.default_rng(0)
    h, w, t, c = 37, 50, 2, 3  # deliberately not multiples of 32
    x = rng.normal(size=(t, c, h, w)).astype(np.float32)
    y = (rng.random((h, w)) > 0.9).astype(np.uint8)
    valid = rng.random((h, w)) > 0.1
    sample = TemporalSample(
        event_id="synthetic_event",
        input_dates=(date(2020, 8, 1), date(2020, 8, 2)),
        target_date=date(2020, 8, 3),
        x=x,
        y=y,
        valid=valid,
        channels=("a", "b", "c"),
        transform=(375.0, 0.0, 0.0, 0.0, -375.0, 0.0),
        crs="EPSG:32610",
    )
    batch = pad_samples([sample], multiple=32)

    x2, info = pad_to_multiple(torch.from_numpy(x), 32)
    m2, _ = pad_valid_mask(torch.from_numpy(valid), 32)

    assert (info.padded_height, info.padded_width) == (64, 64)
    assert torch.equal(batch.x[0], x2)
    assert torch.equal(batch.valid[0, 0], m2)
    assert not batch.valid[0, 0, h:, :].any() and not batch.valid[0, 0, :, w:].any()
