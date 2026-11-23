"""Deterministic, target-blind spatial OOF inference and compact NPZ export."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fireguard_ml.data.event_splits import read_manifest, sha256_file, validate_manifest
from fireguard_ml.data.sample_adapter import adapt_inputs_for_inference
from fireguard_ml.data.sample_adapter import restore_frame as restore_input_frame
from fireguard_ml.models.checkpointing import load_spatial_checkpoint
from fireguard_ml.training.config import configure_reproducibility


def resolve_oof_assignment(
    event_id: str,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> tuple[int, str, list]:
    """Return the unique held-out fold/group for an event after verifying the frozen manifest."""
    path = Path(manifest_path)
    actual = sha256_file(path)
    if actual.lower() != expected_manifest_sha256.lower():
        raise ValueError("P1 manifest SHA-256 does not match the release manifest")
    rows = read_manifest(path)
    validate_manifest(rows)
    tests = [row for row in rows if row.event_id == event_id and row.split == "test"]
    if len(tests) != 1:
        raise ValueError(f"Event {event_id!r} must be held out in exactly one P1 fold")
    row = tests[0]
    return row.fold_test_year, row.group_id, rows


def validate_oof_checkpoint_metadata(
    meta: dict[str, Any],
    *,
    fold_test_year: int,
    held_out_group_id: str,
    rows: list,
    expected_manifest_sha256: str,
    selected_channels: list[str],
    model_version: str,
    data_contract_version: str,
) -> None:
    """Require checkpoint provenance to match the fold and exclude its test group."""
    if int(meta["test_fold_year"]) != fold_test_year:
        raise ValueError("Checkpoint was not trained for the event's held-out fold")
    if meta["split_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Checkpoint split hash differs from the release split hash")
    if meta["strict"] is not True:
        raise ValueError("Only a strict checkpoint can enter the strict OOF release")
    if meta["model_version"] != model_version:
        raise ValueError("Checkpoint model version differs from release manifest")
    if meta["data_contract_version"] != data_contract_version:
        raise ValueError("Checkpoint data contract differs from release manifest")
    if list(meta["channel_names"]) != selected_channels:
        raise ValueError("Checkpoint channel order differs from release manifest")

    fold_rows = [row for row in rows if row.fold_test_year == fold_test_year]
    train_groups = {row.group_id for row in fold_rows if row.split == "train"}
    validation_groups = {row.group_id for row in fold_rows if row.split == "validation"}
    test_groups = {row.group_id for row in fold_rows if row.split == "test"}
    recorded_train = set(meta["training_group_ids"])
    recorded_validation = set(meta["validation_group_ids"])
    if recorded_train != train_groups:
        raise ValueError("Checkpoint training groups do not match the frozen fold manifest")
    if recorded_validation != validation_groups:
        raise ValueError("Checkpoint validation groups do not match the frozen fold manifest")
    if (
        held_out_group_id not in test_groups
        or held_out_group_id in recorded_train | recorded_validation
    ):
        raise ValueError("Held-out event group appears in checkpoint training/validation metadata")


def _read_inputs_only(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `x` and its source mask only; never access a target member if one is present."""
    source = Path(path)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
            if not {"x.npy", "input_valid_mask.npy"}.issubset(names):
                raise ValueError("Input archive must contain x.npy and input_valid_mask.npy")
            x = np.load(io.BytesIO(archive.read("x.npy")), allow_pickle=False)
            mask = np.load(io.BytesIO(archive.read("input_valid_mask.npy")), allow_pickle=False)
        return x, mask
    with np.load(source, allow_pickle=False) as archive:
        if not {"x", "input_valid_mask"}.issubset(archive.files):
            raise ValueError("Input NPZ must contain x and input_valid_mask")
        return archive["x"].copy(), archive["input_valid_mask"].copy()


def run_spatial_oof_inference(
    *,
    event_id: str,
    sample_id: str,
    input_archive: str | Path,
    release_manifest_path: str | Path,
    project_root: str | Path,
    output_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Choose the held-out checkpoint for the event, infer without target access, export."""
    root = Path(project_root)
    release_path = Path(release_manifest_path)
    if not release_path.is_absolute():
        release_path = root / release_path
    release = json.loads(release_path.read_text(encoding="utf-8"))
    split_path = Path(release["split_manifest_path"])
    if not split_path.is_absolute():
        split_path = root / split_path
    expected_split_hash = str(release["split_manifest_sha256"])
    fold, group_id, rows = resolve_oof_assignment(event_id, split_path, expected_split_hash)

    fold_entry = release["checkpoints"].get(str(fold))
    if not fold_entry:
        raise ValueError(f"No released checkpoint is registered for test fold {fold}")
    checkpoint_path = Path(fold_entry["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256.lower() != str(fold_entry["sha256"]).lower():
        raise ValueError("Checkpoint SHA-256 does not match the release manifest")
    model, meta = load_spatial_checkpoint(checkpoint_path, map_location=device)
    validate_oof_checkpoint_metadata(
        meta,
        fold_test_year=fold,
        held_out_group_id=group_id,
        rows=rows,
        expected_manifest_sha256=expected_split_hash,
        selected_channels=list(release["selected_channels"]),
        model_version=release["model_version"],
        data_contract_version=release["data_contract_version"],
    )
    if meta["config_sha256"] != release["config_sha256"]:
        raise ValueError("Checkpoint config hash differs from release manifest")

    x_np, input_mask_np = _read_inputs_only(input_archive)
    sample = adapt_inputs_for_inference(
        torch.from_numpy(x_np.copy()),
        channel_names=tuple(release["input_channel_names"]),
        selected_channels=tuple(release["selected_channels"]),
        strict_approved=True,
        approval_reference=release["approval_reference"],
        source_input_valid_mask=torch.from_numpy(input_mask_np.copy()),
        pad_multiple=int(release.get("pad_multiple", 32)),
    )
    model = model.to(device).eval()
    configure_reproducibility(int(meta["seed"]), deterministic=True)
    with torch.inference_mode():
        logits = model(sample.inputs.to(device))
        probs_padded = torch.sigmoid(logits)
    probabilities = restore_input_frame(probs_padded, sample).cpu().numpy().astype(np.float32)
    valid_mask = restore_input_frame(sample.input_valid_mask, sample).cpu().numpy().astype(bool)
    probabilities = np.where(valid_mask, probabilities, np.nan).astype(np.float32)

    metadata = {
        "kind": "SPATIAL_OOF_PROBABILITY_EXPORT_NOT_TARGET",
        "event_id": event_id,
        "sample_id": sample_id,
        "event_group_id": group_id,
        "test_fold_year": fold,
        "model_version": release["model_version"],
        "config_sha256": release["config_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": expected_split_hash,
        "data_contract_version": release["data_contract_version"],
        "channel_names": list(release["selected_channels"]),
        "validity_status": sample.validity_status,
        "output_shape": list(probabilities.shape),
        "invalid_probability_encoding": "NaN; use valid_mask for nodata",
        "target_accessed": False,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            probabilities=probabilities,
            valid_mask=valid_mask.astype(np.uint8),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    return metadata
