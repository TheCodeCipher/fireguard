import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fireguard_ml.data.event_splits import sha256_file
from fireguard_ml.data.padding import unpad
from fireguard_ml.data.sample_adapter import adapt_inputs_for_inference
from fireguard_ml.inference.spatial import (
    resolve_oof_assignment,
    run_spatial_oof_inference,
    validate_oof_checkpoint_metadata,
)
from fireguard_ml.models.checkpointing import (
    load_spatial_checkpoint,
    save_spatial_checkpoint,
)
from fireguard_ml.models.unet import UNet
from fireguard_ml.training.losses import masked_bce_dice_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
P1_MANIFEST = REPO_ROOT / "data/manifests/p1_loyo_fold_manifest.csv"
EVENT_ID = "fire_21458798"
CHANNELS = ["feature_a", "feature_c"]
INPUT_CHANNELS = ["feature_a", "feature_b", "feature_c"]


def _release_fixture(tmp_path):
    split_hash = sha256_file(P1_MANIFEST)
    fold, group, rows = resolve_oof_assignment(EVENT_ID, P1_MANIFEST, split_hash)
    fold_rows = [r for r in rows if r.fold_test_year == fold]
    train_groups = sorted({r.group_id for r in fold_rows if r.split == "train"})
    validation_groups = sorted({r.group_id for r in fold_rows if r.split == "validation"})
    torch.manual_seed(62)
    model = UNet(in_channels=2, base_channels=2)
    meta = {
        "model_version": "synthetic-parity-test-only",
        "channel_names": CHANNELS,
        "data_contract_version": "synthetic-fixture-v1",
        "config_sha256": "a" * 64,
        "split_manifest_sha256": split_hash,
        "test_fold_year": fold,
        "seed": 62,
        "environment": {"python": "test", "device": "cpu"},
        "training_group_ids": train_groups,
        "validation_group_ids": validation_groups,
        "strict": True,
    }
    checkpoint = tmp_path / "synthetic-checkpoint.pt"
    save_spatial_checkpoint(model, checkpoint, meta)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    release = {
        "model_version": meta["model_version"],
        "data_contract_version": meta["data_contract_version"],
        "config_sha256": meta["config_sha256"],
        "split_manifest_path": str(P1_MANIFEST),
        "split_manifest_sha256": split_hash,
        "input_channel_names": INPUT_CHANNELS,
        "selected_channels": CHANNELS,
        "approval_reference": "synthetic-test-fixture-only",
        "pad_multiple": 32,
        "checkpoints": {str(fold): {"path": str(checkpoint), "sha256": checkpoint_hash}},
    }
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    x = np.random.default_rng(5).normal(size=(1, 3, 304, 247)).astype(np.float32)
    mask = np.ones((1, 1, 304, 247), dtype=np.uint8)
    mask[0, 0, 25, 31] = 0
    x[0, 0, 25, 31] = np.nan
    input_path = tmp_path / "inputs-with-unread-target.npz"
    # A deliberately non-label target member proves the inference path reads x and mask only.
    np.savez_compressed(input_path, x=x, input_valid_mask=mask, y=np.asarray(["DO_NOT_READ"]))
    return model, meta, release, release_path, input_path, fold, group, rows


def test_event_resolves_to_unique_held_out_fold_and_rejects_wrong_hash():
    split_hash = sha256_file(P1_MANIFEST)
    fold, group, rows = resolve_oof_assignment(EVENT_ID, P1_MANIFEST, split_hash)
    assert fold == 2018
    assert group
    assert rows
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_oof_assignment(EVENT_ID, P1_MANIFEST, "0" * 64)


@pytest.mark.parametrize(
    ("height", "width", "padded_hw"),
    [(304, 247, (320, 256)), (305, 207, (320, 224))],
)
def test_deterministic_full_frame_inference_multiple_native_sizes(height, width, padded_hw):
    torch.manual_seed(88)
    model = UNet(in_channels=2, base_channels=2).eval()
    x = torch.randn(1, 3, height, width)
    valid = torch.ones(1, 1, height, width, dtype=torch.bool)
    sample = adapt_inputs_for_inference(
        x,
        channel_names=tuple(INPUT_CHANNELS),
        selected_channels=tuple(CHANNELS),
        source_input_valid_mask=valid,
    )
    assert sample.inputs.shape[-2:] == padded_hw
    assert not sample.input_valid_mask[..., height:, :].any()
    assert not sample.input_valid_mask[..., :, width:].any()
    with torch.inference_mode():
        first = torch.sigmoid(model(sample.inputs))
        second = torch.sigmoid(model(sample.inputs))
    assert torch.equal(first, second)
    assert unpad(first, sample.pad_info).shape[-2:] == (height, width)


def test_oof_checkpoint_rejects_held_out_group_in_training(tmp_path):
    _, meta, release, _, _, fold, group, rows = _release_fixture(tmp_path)
    meta["training_group_ids"] = [*meta["training_group_ids"], group]
    with pytest.raises(ValueError, match="training groups"):
        validate_oof_checkpoint_metadata(
            meta,
            fold_test_year=fold,
            held_out_group_id=group,
            rows=rows,
            expected_manifest_sha256=release["split_manifest_sha256"],
            selected_channels=CHANNELS,
            model_version=release["model_version"],
            data_contract_version=release["data_contract_version"],
        )


def test_masked_loss_is_finite_for_rare_positive_and_bce_dice_weight_sensitivity():
    targets = torch.zeros(1, 1, 1, 2048)
    targets[..., 17] = 1
    logits = torch.zeros_like(targets, requires_grad=True)
    valid = torch.ones_like(targets, dtype=torch.bool)
    combined = masked_bce_dice_loss(logits, targets, valid)
    bce_only = masked_bce_dice_loss(logits, targets, valid, bce_weight=1, dice_weight=0)
    dice_only = masked_bce_dice_loss(logits, targets, valid, bce_weight=0, dice_weight=1)
    assert torch.isfinite(combined) and torch.isfinite(bce_only) and torch.isfinite(dice_only)
    assert not torch.equal(combined, bce_only)
    combined.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_spatial_checkpoint_round_trip_and_offline_export_parity(tmp_path):
    original_model, _, release, release_path, input_path, _, _, _ = _release_fixture(tmp_path)
    output_path = tmp_path / "oof" / "probabilities.npz"
    run_meta = run_spatial_oof_inference(
        event_id=EVENT_ID,
        sample_id="synthetic-origin-2018-01-06",
        input_archive=input_path,
        release_manifest_path=release_path,
        project_root=REPO_ROOT,
        output_path=output_path,
    )
    assert output_path.is_file()
    assert run_meta["target_accessed"] is False
    assert run_meta["test_fold_year"] == 2018
    assert run_meta["output_shape"] == [1, 1, 304, 247]

    with np.load(output_path, allow_pickle=False) as exported:
        probs = exported["probabilities"]
        valid = exported["valid_mask"].astype(bool)
        export_meta = json.loads(str(exported["metadata_json"]))
    assert probs.shape == valid.shape == (1, 1, 304, 247)
    assert np.isnan(probs[~valid]).all()
    assert np.isfinite(probs[valid]).all()
    assert np.all((probs[valid] >= 0) & (probs[valid] <= 1))
    assert export_meta["target_accessed"] is False
    assert "target" not in export_meta

    # Compare the saved artifact against the in-memory model using identical masking/padding.
    x_np = np.load(input_path, allow_pickle=False)["x"]
    mask_np = np.load(input_path, allow_pickle=False)["input_valid_mask"]
    prepared = adapt_inputs_for_inference(
        torch.from_numpy(x_np),
        channel_names=tuple(INPUT_CHANNELS),
        selected_channels=tuple(CHANNELS),
        strict_approved=True,
        approval_reference=release["approval_reference"],
        source_input_valid_mask=torch.from_numpy(mask_np),
    )
    original_model.eval()
    with torch.inference_mode():
        direct = torch.sigmoid(original_model(prepared.inputs))
    direct = unpad(direct, prepared.pad_info).numpy()
    np.testing.assert_allclose(probs[valid], direct[valid], rtol=0.0, atol=1e-7)

    checkpoint_path = Path(release["checkpoints"]["2018"]["path"])
    loaded, loaded_meta = load_spatial_checkpoint(checkpoint_path)
    assert loaded_meta["model_version"] == release["model_version"]
    assert not loaded.training
    with torch.inference_mode():
        loaded_probs = torch.sigmoid(loaded(prepared.inputs)).numpy()
    loaded_probs = unpad(torch.from_numpy(loaded_probs), prepared.pad_info).numpy()
    np.testing.assert_allclose(direct[valid], loaded_probs[valid], rtol=0.0, atol=1e-7)


def test_offline_export_cli_runs_synthetic_target_blind_export(tmp_path):
    _, _, _, release_path, input_path, _, _, _ = _release_fixture(tmp_path)
    script = REPO_ROOT / "scripts/export_spatial_oof.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--event-id",
            EVENT_ID,
            "--sample-id",
            "synthetic-cli-check",
            "--input-archive",
            str(input_path),
            "--release-manifest",
            str(release_path),
            "--project-root",
            str(REPO_ROOT),
            "--output",
            str(tmp_path / "cli-output.npz"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_metadata = json.loads(result.stdout)
    assert cli_metadata["target_accessed"] is False
    assert cli_metadata["sample_id"] == "synthetic-cli-check"
    assert (tmp_path / "cli-output.npz").is_file()
