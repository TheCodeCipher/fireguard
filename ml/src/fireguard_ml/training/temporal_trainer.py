"""Fold-wise training of the temporal models M3 / M4 under the frozen P1 protocol (Role 3).

One call trains one (model, context length, seed, fold) and writes, under ``out_dir/fold_<Y>/``:

* ``<run>.pt``               self-describing checkpoint (models/checkpointing.py), strict=True
* ``<run>.normalizer.json``  train-groups-only statistics
* ``<run>.selection.json``   validation F1 threshold + Platt calibrator (Role 4 isolation API)
* ``<run>.log.jsonl``        one line per epoch
* ``<run>.summary.json``     config/split/env hashes, hardware, time, peak memory, best epoch

Rules enforced here (Phase 3 guide 1.4 and P3-R3-01/02):

* train / validation / test events come only from the frozen manifest, which is validated and
  hash-checked before anything is loaded;
* every sample is an exact row of ``wsts_eval_samples.csv`` (identical to M1's);
* normalisation uses train groups only; thresholds, calibrators and early stopping use
  validation groups only; the test split is never opened here;
* loss and metrics see only valid, unpadded cells; frames are never cropped;
* channels are checked against the leakage register before training starts.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fireguard_ml.data.channel_policy import assert_strict_channels, load_register
from fireguard_ml.data.event_splits import (
    fold_events,
    read_manifest,
    sha256_file,
    validate_manifest,
)
from fireguard_ml.evaluation.calibration import fit_platt, select_f1_threshold
from fireguard_ml.evaluation.metrics import StreamingScoreHistogram, summarise
from fireguard_ml.models.checkpointing import save_checkpoint
from fireguard_ml.models.temporal_unet import MODEL_REGISTRY
from fireguard_ml.training.normalization import FoldNormalizer, fit_normalizer
from fireguard_ml.training.temporal_data import (
    TemporalSampleSource,
    pad_samples,
    scored_keys,
    unpad,
)

PREPROCESSING_VERSION_PLACEHOLDER = "UNSET"


# ------------------------------------------------------------------------------- loss


def _role3_masked_bce_dice(
    logits, targets, valid_mask, *, bce_weight=1.0, dice_weight=1.0, smooth=1.0
):
    """Verbatim copy of Role 2's ``masked_bce_dice_loss`` (FireGuard_Role2_Starter, training/
    losses.py) so M3/M4 and M1 optimise the same objective. Used only while Role 2's module is
    not installed; ``test_temporal_pipeline`` asserts both give identical values."""
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets, and valid_mask must have identical shapes")
    if not torch.is_floating_point(targets):
        targets = targets.float()
    mask = valid_mask.to(device=logits.device, dtype=logits.dtype)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("valid_mask contains no valid pixels")
    bce_map = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    bce = (bce_map * mask).sum() / count
    probs = torch.sigmoid(logits) * mask
    y = targets.to(logits.dtype) * mask
    dice = 1.0 - (2.0 * (probs * y).sum() + smooth) / (probs.sum() + y.sum() + smooth)
    return bce_weight * bce + dice_weight * dice


def resolve_loss():
    """Role 2's shared loss when it is installed, otherwise the verbatim copy above."""
    try:
        from fireguard_ml.training.losses import masked_bce_dice_loss  # Role 2

        return masked_bce_dice_loss, "role2:fireguard_ml.training.losses"
    except ImportError:
        return _role3_masked_bce_dice, "role3:verbatim_copy_of_role2_starter"


# ------------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TemporalRunConfig:
    model: str  # "m3_stacked_unet" | "m4_convlstm_unet"
    T: int
    channels: tuple[str, ...]
    base: int = 32
    depth: int = 5
    lstm_kernel: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    patience: int = 3
    grad_accum: int = 4
    amp: bool = True
    seed: int = 0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    train_samples_per_epoch: int | None = None  # None = every training sample once per epoch
    preprocessing_version: str = PREPROCESSING_VERSION_PLACEHOLDER
    extra: dict = field(default_factory=dict)

    def model_kwargs(self) -> dict:
        kw = dict(c_strict=len(self.channels), T=self.T, base=self.base, depth=self.depth)
        if self.model == "m4_convlstm_unet":
            kw["lstm_kernel"] = self.lstm_kernel
        return kw

    def run_name(self) -> str:
        width = "" if self.base == 32 else f"_base{self.base}"
        return f"{self.model}{width}_T{self.T}_seed{self.seed}"

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ProtocolFiles:
    manifest: Path
    hashes: Path
    eval_samples: Path
    register: Path

    def verified_manifest_hash(self) -> str:
        """Validate the manifest, check it against p1_split_hashes.txt, require FROZEN."""
        text = Path(self.hashes).read_text(encoding="utf-8")
        if "status=FROZEN" not in text:
            raise RuntimeError("P1 split hashes are not FROZEN; real training is not authorised")
        recorded = dict(
            reversed(line.split("  ", 1))
            for line in text.splitlines()
            if line and not line.startswith("#")
        )
        actual = sha256_file(self.manifest)
        if recorded.get(Path(self.manifest).name) != actual:
            raise RuntimeError("p1_loyo_fold_manifest.csv does not match its recorded hash")
        validate_manifest(read_manifest(self.manifest))
        return actual


# ------------------------------------------------------------------------------- helpers


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: TemporalRunConfig) -> torch.nn.Module:
    return MODEL_REGISTRY[cfg.model](**cfg.model_kwargs())


def _batch_tensors(source, norm: FoldNormalizer, keys, device, depth: int):
    samples = []
    for k in keys:
        s = source.load(k)
        samples.append(replace(s, x=norm.apply(s.x, s.valid)))
    b = pad_samples(samples, multiple=2**depth)
    return b, b.x.to(device), b.y.to(device), b.valid.to(device)


@torch.no_grad()
def score_split(model, source, norm, keys, device, depth: int, amp: bool) -> dict:
    """Full-frame probabilities for ``keys``: pooled histogram + per-event histograms."""
    model.eval()
    pooled = StreamingScoreHistogram()
    per_event: dict[str, StreamingScoreHistogram] = {}
    for k in keys:
        b, x, y, m = _batch_tensors(source, norm, [k], device, depth)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        p = torch.sigmoid(logits.float())
        hw = b.orig_hw[0]
        pn = unpad(p, hw)[0, 0].cpu().numpy()
        yn = unpad(y, hw)[0, 0].cpu().numpy().astype(np.uint8)
        mn = unpad(m, hw)[0, 0].cpu().numpy()
        pooled.add(yn, pn, mn)
        per_event.setdefault(k[0], StreamingScoreHistogram(n_bins=2**16)).add(yn, pn, mn)
    return {"pooled": pooled, "per_event": per_event}


def event_blocked_order(keys, rng: random.Random) -> list:
    """Shuffle events, then the samples inside each event.

    Every input frame is shared by T consecutive samples of its event. Visiting an event's
    samples together lets the source's frame cache serve them, so each GeoTIFF is read about once
    per epoch instead of T times (measured: 0.34 s per uncached 5-day sample on CPU). Declared
    before any training; seeded, so runs stay reproducible.
    """
    by_event: dict[str, list] = {}
    for k in keys:
        by_event.setdefault(k[0], []).append(k)
    events = sorted(by_event)
    rng.shuffle(events)
    order = []
    for ev in events:
        ks = by_event[ev][:]
        rng.shuffle(ks)
        order.extend(ks)
    return order


# ------------------------------------------------------------------------------- training


def train_fold(
    cfg: TemporalRunConfig,
    source: TemporalSampleSource,
    protocol: ProtocolFiles,
    fold: int,
    out_dir: Path,
    device: str | torch.device | None = None,
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split_hash = protocol.verified_manifest_hash()
    if source.sequence_length != cfg.T:
        raise ValueError(f"source T={source.sequence_length} but config T={cfg.T}")
    if tuple(source.channels) != tuple(cfg.channels):
        raise ValueError("source channel order differs from the run config")
    assert_strict_channels(list(cfg.channels), load_register(protocol.register))
    if cfg.preprocessing_version == PREPROCESSING_VERSION_PLACEHOLDER and not source.is_mock:
        raise RuntimeError("set preprocessing_version to Role 1's loader/contract version")

    rows = read_manifest(protocol.manifest)
    train_ev = set(fold_events(rows, fold, "train"))
    val_ev = set(fold_events(rows, fold, "validation"))
    available = set(source.keys())
    train_keys = [k for k in scored_keys(protocol.eval_samples, train_ev) if k in available]
    val_keys = [k for k in scored_keys(protocol.eval_samples, val_ev) if k in available]
    if not train_keys or not val_keys:
        raise RuntimeError(f"fold {fold}: no train or validation samples available from source")

    seed_everything(cfg.seed)
    loss_fn, loss_impl = resolve_loss()
    # PROTOTYPE: progress lines, so a multi-hour real run is visibly alive. The first pass reads
    # every training frame once to fit the per-fold normaliser (no model work yet).
    print(f"[{cfg.run_name()} fold {fold}] {len(train_keys)} train / {len(val_keys)} validation "
          f"samples on {device}; fitting normaliser on the training frames...", flush=True)
    norm = fit_normalizer(source, train_keys, train_ev, fold, split_hash)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run = cfg.run_name()
    fdir = Path(out_dir) / f"fold_{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    log_path = fdir / f"{run}.log.jsonl"
    log_path.write_text("", encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rng = random.Random(cfg.seed)
    best = {"auprc": -1.0, "epoch": -1, "state": None, "val": None}
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        order = event_blocked_order(train_keys, rng)
        if cfg.train_samples_per_epoch:
            order = order[: cfg.train_samples_per_epoch]
        opt.zero_grad(set_to_none=True)
        losses, skipped = [], 0
        for i, k in enumerate(order, 1):
            _, x, y, m = _batch_tensors(source, norm, [k], device, cfg.depth)
            if not m.any():
                skipped += 1
                continue
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(
                logits.float(), y, m, bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight
            )
            scaler.scale(loss / cfg.grad_accum).backward()
            losses.append(float(loss.detach()))
            if i % cfg.grad_accum == 0 or i == len(order):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if i % 500 == 0 and losses:
                print(f"  epoch {epoch}: {i}/{len(order)} samples, "
                      f"loss {np.mean(losses[-500:]):.4f}, {time.time() - t0:.0f}s", flush=True)
        val = score_split(model, source, norm, val_keys, device, cfg.depth, cfg.amp)
        val_sum = summarise(val["pooled"].counts())
        rec = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "train_samples": len(losses),
            "skipped_no_valid": skipped,
            "val_auprc": val_sum["auprc"],
            "val_brier": val_sum["brier"],
            "val_n_pos": val_sum["n_pos"],
            "seconds": round(time.time() - t0, 1),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  epoch {epoch} done: {json.dumps(rec)}", flush=True)
        if val_sum["auprc"] > best["auprc"]:
            best = {
                "auprc": val_sum["auprc"],
                "epoch": epoch,
                "val": val,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best["state"])
    val_counts = best["val"]["pooled"].counts()
    threshold = select_f1_threshold(val_counts, "P1", fold, "validation")
    calibrator = fit_platt(val_counts, "P1", fold, "validation")

    meta = {
        "channels": list(cfg.channels),
        "T": cfg.T,
        "preprocessing_version": cfg.preprocessing_version,
        "config_hash": cfg.config_hash(),
        "fold": fold,
        "split_manifest_hash": split_hash,
        "protocol": "P1_LOYO",
        "best_epoch": best["epoch"],
        "val_auprc": best["auprc"],
        "seed": cfg.seed,
        "is_mock": bool(source.is_mock),
        "loss_impl": loss_impl,
    }
    ckpt = fdir / f"{run}.pt"
    save_checkpoint(model.cpu(), ckpt, meta, strict=True)
    norm.to_json(fdir / f"{run}.normalizer.json")
    selection = {
        "protocol": "P1",
        "fold": fold,
        "fitted_on": "validation",
        "threshold": asdict(threshold),
        "calibrator": asdict(calibrator),
        "calibration_note": "Platt scaling of model probability; uncertainty layer NOT produced",
    }
    (fdir / f"{run}.selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    summary = {
        "run": run,
        "fold": fold,
        "is_mock": bool(source.is_mock),
        "config": asdict(cfg),
        "config_hash": cfg.config_hash(),
        "split_manifest_hash": split_hash,
        "checkpoint": ckpt.name,
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "best_epoch": best["epoch"],
        "val_auprc": best["auprc"],
        "train_samples": len(train_keys),
        "val_samples": len(val_keys),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.time() - t0, 1),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "loss_impl": loss_impl,
    }
    (fdir / f"{run}.summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
