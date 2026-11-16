"""Train one model on one leave-one-year-out (P1) fold, resumably.

``train_fold`` writes, under ``<out_dir>/fold_<Y>/`` (``<run>`` = model, width, T and seed):

    while training (deleted when the fold finishes)
        <run>.last.pt            full training state after the last finished epoch (resume point)
        <run>.best_weights.pt    weights of the best epoch so far
    when the fold is finished
        <run>.pt                 the best model: self-describing strict checkpoint (inference)
        <run>.normalizer.json    per-channel statistics from this fold's training groups
        <run>.selection.json     F1 threshold + Platt calibrator fitted on validation groups
        <run>.history.csv        one row per epoch (loss, learning rate, validation metrics)
        <run>.summary.json       config, hashes, hardware, best epoch, time

Interrupt it at any point and call it again with the same config: it continues from the last
finished epoch (same optimizer, schedule, early-stopping counter and random state). A fold with a
summary for the same config hash is skipped.

How a fold is trained
    * Data: full event frames (no cropping), shuffled across events every epoch so consecutive
      updates are not dominated by one fire. Frames are padded to a multiple of 32 and padding is
      masked out of the loss and of every metric.
    * Augmentation (``augment_flips``): random horizontal/vertical flips. Aspect is stored as
      sin(aspect) (east-west component), so a horizontal flip negates it and a vertical flip
      leaves it unchanged; every other channel is direction-free. Rotations are not used because
      cos(aspect) is not available to rotate correctly.
    * Loss: masked BCE + soft Dice (plan v4 4.6), Dice taken over the whole batch.
    * Optimiser: AdamW, linear warm-up then cosine decay of the learning rate, gradient clipping,
      mixed precision where the GPU supports it (see training/hardware.py).
    * Model selection: validation AUPRC on full frames at natural prevalence after every epoch;
      early stopping after ``patience`` epochs without an improvement of at least ``min_delta``.
    * Rules (Phase 3 guide 1.4): splits come only from the frozen manifest (hash-checked);
      normalisation uses training groups only; thresholds, calibrators and early stopping use
      validation groups only; the test split is never opened here; channels are checked against
      the leakage register before anything is loaded.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import os
import platform
import queue
import random
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from fireguard_ml.data.channel_policy import assert_strict_channels, load_register
from fireguard_ml.data.event_splits import (
    fold_events,
    read_manifest,
    sha256_file,
    validate_manifest,
)
from fireguard_ml.evaluation.calibration import fit_platt, select_f1_threshold
from fireguard_ml.evaluation.metrics import StreamingScoreHistogram, summarise
from fireguard_ml.models.architectures import MODEL_REGISTRY
from fireguard_ml.models.checkpointing import save_checkpoint
from fireguard_ml.training.hardware import Hardware, pick_device
from fireguard_ml.training.losses import masked_bce_dice_loss
from fireguard_ml.training.normalization import FoldNormalizer, fit_normalizer
from fireguard_ml.training.temporal_data import (
    TemporalSampleSource,
    pad_samples,
    scored_keys,
    unpad,
)

PREPROCESSING_VERSION_PLACEHOLDER = "UNSET"
LOSS_IMPL = "fireguard_ml.training.losses.masked_bce_dice_loss"


# ------------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TemporalRunConfig:
    """Everything that defines a training run (and therefore its config hash).

    Hardware choices (batch size, workers, precision) are deliberately NOT here: they change
    speed, not the experiment, because gradient accumulation keeps ``effective_batch`` fixed.
    """

    model: str
    T: int
    channels: tuple[str, ...]
    base: int = 32
    depth: int = 5
    lstm_kernel: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_epochs: float = 1.0
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 6
    min_delta: float = 1e-4
    effective_batch: int = 16
    grad_clip: float = 1.0
    augment_flips: bool = True
    amp: bool = True
    seed: int = 0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    train_samples_per_epoch: int | None = None  # None = every training sample once per epoch
    tag: str = ""  # names an ablation, e.g. "no_terrain"
    preprocessing_version: str = PREPROCESSING_VERSION_PLACEHOLDER
    extra: dict = field(default_factory=dict)

    def model_kwargs(self) -> dict:
        kw = dict(c_strict=len(self.channels), T=self.T, base=self.base, depth=self.depth,
                  dropout=self.dropout)
        if self.model == "m4_convlstm_unet":
            kw["lstm_kernel"] = self.lstm_kernel
        return kw

    def run_name(self) -> str:
        width = "" if self.base == 32 else f"_base{self.base}"
        tag = f"_{self.tag}" if self.tag else ""
        return f"{self.model}{width}_T{self.T}{tag}_seed{self.seed}"

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


def flip_sample(x: np.ndarray, y: np.ndarray, channels, horizontal: bool, vertical: bool):
    """Flip one sample ``x`` [T, C, H, W] and ``y`` [H, W] consistently (see module docstring)."""
    if horizontal:
        x, y = x[..., ::-1], y[..., ::-1]
        if "aspect" in channels:  # sin(aspect) is the east-west component: mirrored E<->W
            x = x.copy()
            x[:, list(channels).index("aspect")] *= -1.0
    if vertical:
        x, y = x[..., ::-1, :], y[..., ::-1, :]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class FoldDataset(Dataset):
    """Normalised training samples for one fold; flips applied on the fly."""

    def __init__(self, source, norm: FoldNormalizer, keys, augment: bool):
        self.source, self.norm, self.keys, self.augment = source, norm, list(keys), augment

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        s = self.source.load(self.keys[i])
        x = self.norm.apply(s.x, s.valid)
        y = s.y.astype(np.float32)
        if self.augment:
            h, v = (torch.rand(2) < 0.5).tolist()
            x, y = flip_sample(x, y, s.channels, h, v)
        return x, y, s.valid


class EpochOrder(Sampler):
    """Sample order set by the trainer before every epoch (lets loader workers persist)."""

    def __init__(self) -> None:
        self.order: list[int] = []

    def __iter__(self):
        return iter(list(self.order))

    def __len__(self) -> int:
        return len(self.order)


def pad_collate(items, multiple: int = 32):
    """Pad a list of (x, y, valid) to a common size, a multiple of ``multiple`` (mask = False)."""
    hmax = max(it[0].shape[-2] for it in items)
    wmax = max(it[0].shape[-1] for it in items)
    hp, wp = hmax + (-hmax) % multiple, wmax + (-wmax) % multiple
    t, c = items[0][0].shape[:2]
    xb = torch.zeros(len(items), t, c, hp, wp)
    yb = torch.zeros(len(items), 1, hp, wp)
    mb = torch.zeros(len(items), 1, hp, wp, dtype=torch.bool)
    for i, (x, y, m) in enumerate(items):
        h, w = x.shape[-2:]
        xb[i, :, :, :h, :w] = torch.from_numpy(x)
        yb[i, 0, :h, :w] = torch.from_numpy(y)
        mb[i, 0, :h, :w] = torch.from_numpy(m)
    return xb, yb, mb


def _pad_batch_cpu(source, norm: FoldNormalizer, keys, depth: int):
    samples = []
    for k in keys:
        s = source.load(k)
        samples.append(replace(s, x=norm.apply(s.x, s.valid)))
    return pad_samples(samples, multiple=2**depth)


def _iter_padded_batches(source, norm: FoldNormalizer, keys, depth: int, prefetch: int = 4):
    """Background-thread producer so loading overlaps with GPU compute (evaluation path)."""
    if len(keys) <= 2:
        for k in keys:
            yield k, _pad_batch_cpu(source, norm, [k], depth)
        return

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def _producer():
        try:
            for k in keys:
                if stop.is_set():
                    break
                q.put((k, _pad_batch_cpu(source, norm, [k], depth), None))
        except BaseException as exc:
            q.put((None, None, exc))
        finally:
            q.put((None, None, StopIteration))

    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    try:
        while True:
            k, b, err = q.get()
            if err is StopIteration:
                break
            if err is not None:
                raise err
            yield k, b
    finally:
        stop.set()


@torch.no_grad()
def score_split(model, source, norm, keys, device, depth: int, amp=False) -> dict:
    """Full-frame probabilities for ``keys``: pooled histogram + per-event histograms.

    ``amp``: False, True (float16) or a torch dtype for autocast on CUDA.
    """
    device = torch.device(device)
    dtype = amp if isinstance(amp, torch.dtype) else torch.float16
    use_amp = bool(amp) and device.type == "cuda"
    model.eval()
    pooled = StreamingScoreHistogram()
    per_event: dict[str, StreamingScoreHistogram] = {}
    for k, b in _iter_padded_batches(source, norm, keys, depth):
        x = b.x.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            logits = model(x)
        p = torch.sigmoid(logits.float())
        hw = b.orig_hw[0]
        pn = unpad(p, hw)[0, 0].cpu().numpy()
        yn = unpad(b.y, hw)[0, 0].numpy().astype(np.uint8)
        mn = unpad(b.valid, hw)[0, 0].numpy()
        pooled.add(yn, pn, mn)
        per_event.setdefault(k[0], StreamingScoreHistogram(n_bins=2**16)).add(yn, pn, mn)
    return {"pooled": pooled, "per_event": per_event}


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warm-up to 1, then cosine decay to ``min_ratio`` at ``total`` steps."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_history(path: Path, history: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in history for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(history)


def default_hardware(device: str | None = None) -> Hardware:
    """Small, safe settings when no hardware plan is given (tests, quick runs)."""
    dev = pick_device(device)
    return Hardware(str(dev), None, 2, 1, 0)


# ------------------------------------------------------------------------------- training


def train_fold(
    cfg: TemporalRunConfig,
    source: TemporalSampleSource,
    protocol: ProtocolFiles,
    fold: int,
    out_dir: Path,
    device: str | torch.device | None = None,
    hw: Hardware | None = None,
    log=print,
) -> dict:
    hw = hw or default_hardware(str(device) if device is not None else None)
    device = torch.device(hw.device)
    split_hash = protocol.verified_manifest_hash()
    if source.sequence_length != cfg.T:
        raise ValueError(f"source T={source.sequence_length} but config T={cfg.T}")
    if tuple(source.channels) != tuple(cfg.channels):
        raise ValueError("source channel order differs from the run config")
    assert_strict_channels(list(cfg.channels), load_register(protocol.register))
    if cfg.preprocessing_version == PREPROCESSING_VERSION_PLACEHOLDER and not source.is_mock:
        raise RuntimeError("set preprocessing_version to Role 1's loader/contract version")

    run = cfg.run_name()
    fdir = Path(out_dir) / f"fold_{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    paths = {k: fdir / f"{run}{suffix}" for k, suffix in {
        "ckpt": ".pt", "norm": ".normalizer.json", "sel": ".selection.json",
        "hist": ".history.csv", "summary": ".summary.json", "last": ".last.pt",
        "best": ".best_weights.pt"}.items()}
    chash = cfg.config_hash()
    if paths["summary"].exists() and paths["ckpt"].exists():
        done = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if done.get("config_hash") == chash:
            log(f"[{run} fold {fold}] already finished (best epoch {done['best_epoch']}, "
                f"validation AUPRC {done['val_auprc']:.4f}); skipping")
            return done
        log(f"[{run} fold {fold}] an older run with a different config exists; retraining")

    rows = read_manifest(protocol.manifest)
    train_ev = set(fold_events(rows, fold, "train"))
    val_ev = set(fold_events(rows, fold, "validation"))
    available = set(source.keys())
    train_keys = [k for k in scored_keys(protocol.eval_samples, train_ev) if k in available]
    val_keys = [k for k in scored_keys(protocol.eval_samples, val_ev) if k in available]
    if not train_keys or not val_keys:
        raise RuntimeError(f"fold {fold}: no train or validation samples available from source")

    seed_everything(cfg.seed)
    state = None
    if paths["last"].exists():
        state = torch.load(paths["last"], map_location="cpu", weights_only=False)
        if state.get("config_hash") != chash:
            log(f"[{run} fold {fold}] resume file is from another config; starting over")
            state = None
    if state is not None and paths["norm"].exists():
        norm = FoldNormalizer.from_json(paths["norm"])
    else:
        log(f"[{run} fold {fold}] {len(train_keys)} train / {len(val_keys)} validation samples; "
            f"fitting the normaliser on training frames...")
        norm = fit_normalizer(source, train_keys, train_ev, fold, split_hash)
        norm.to_json(paths["norm"])

    model = build_model(cfg).to(device)
    decay, no_decay = [], []
    for _, p in model.named_parameters():  # no weight decay on norms and biases (standard)
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    n_train = min(len(train_keys), cfg.train_samples_per_epoch or len(train_keys))
    batches_per_epoch = math.ceil(n_train / hw.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / hw.grad_accum)
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total_steps, warmup, cfg.min_lr / cfg.lr))
    amp_dtype = hw.torch_amp_dtype if (cfg.amp and device.type == "cuda") else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    start_epoch, best_auprc, best_epoch, bad_epochs, history = 0, -1.0, -1, 0, []
    t_prev = 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_auprc, best_epoch = state["best_auprc"], state["best_epoch"]
        bad_epochs, history, t_prev = state["bad_epochs"], state["history"], state["seconds"]
        torch.set_rng_state(state["torch_rng"])
        log(f"[{run} fold {fold}] resuming after epoch {state['epoch']} "
            f"(best so far: epoch {best_epoch}, validation AUPRC {best_auprc:.4f})")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    stopped_early = bad_epochs >= cfg.patience
    t0 = time.time() - t_prev
    sampler = EpochOrder()
    workers = 0 if source.is_mock else hw.num_workers
    loader = DataLoader(FoldDataset(source, norm, train_keys, cfg.augment_flips),
                        batch_size=hw.batch_size, sampler=sampler, num_workers=workers,
                        collate_fn=functools.partial(pad_collate, multiple=2**cfg.depth),
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                        worker_init_fn=_seed_worker)
    for epoch in range(start_epoch, cfg.max_epochs):
        if stopped_early:
            break
        model.train()
        g = torch.Generator().manual_seed(cfg.seed * 1000 + epoch)  # same order after a resume
        sampler.order = torch.randperm(len(train_keys), generator=g).tolist()[:n_train]
        torch.manual_seed(cfg.seed * 1000 + epoch)  # flip draws (in-process loading)
        opt.zero_grad(set_to_none=True)
        losses, seen, te = [], 0, time.time()
        for i, (x, y, m) in enumerate(loader, 1):
            if not m.any():
                continue
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(x)
            loss = masked_bce_dice_loss(logits.float(), y, m, bce_weight=cfg.bce_weight,
                                        dice_weight=cfg.dice_weight)
            scaler.scale(loss / hw.grad_accum).backward()
            losses.append(loss.detach())
            seen += x.shape[0]
            if i % hw.grad_accum == 0 or i == batches_per_epoch:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if i % max(1, 400 // hw.batch_size) == 0:
                log(f"  epoch {epoch}: {seen}/{n_train} samples, loss "
                    f"{float(torch.stack(losses[-50:]).mean()):.4f}, "
                    f"{seen / (time.time() - te):.1f} samples/s")
        val = score_split(model, source, norm, val_keys, device, cfg.depth, amp_dtype or False)
        vs = summarise(val["pooled"].counts())
        rec = {"epoch": epoch,
               "train_loss": float(torch.stack(losses).mean()) if losses else float("nan"),
               "lr": opt.param_groups[0]["lr"], "val_auprc": vs["auprc"],
               "val_brier": vs["brier"], "val_ece": vs["ece"], "val_n_pos": vs["n_pos"],
               "epoch_seconds": round(time.time() - te, 1)}
        history.append(rec)
        improved = vs["auprc"] > best_auprc + cfg.min_delta
        if improved:
            best_auprc, best_epoch, bad_epochs = vs["auprc"], epoch, 0
            _save_atomic({k: v.detach().cpu() for k, v in model.state_dict().items()},
                         paths["best"])
        else:
            bad_epochs += 1
        stopped_early = bad_epochs >= cfg.patience
        _save_atomic({"config_hash": chash, "epoch": epoch, "model": model.state_dict(),
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                      "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                      "best_auprc": best_auprc, "best_epoch": best_epoch,
                      "bad_epochs": bad_epochs, "history": history,
                      "seconds": time.time() - t0, "torch_rng": torch.get_rng_state()},
                     paths["last"])
        _write_history(paths["hist"], history)
        log(f"  epoch {epoch} done: loss {rec['train_loss']:.4f}, val AUPRC {vs['auprc']:.4f}"
            f"{' (best)' if improved else ''}, {rec['epoch_seconds']:.0f}s"
            f"{', early stop' if stopped_early else ''}")

    # ---- finish: best weights -> calibration on validation -> final files
    model.load_state_dict(torch.load(paths["best"], map_location="cpu", weights_only=True))
    model.to(device)
    val = score_split(model, source, norm, val_keys, device, cfg.depth, amp_dtype or False)
    val_counts = val["pooled"].counts()
    threshold = select_f1_threshold(val_counts, "P1", fold, "validation")
    calibrator = fit_platt(val_counts, "P1", fold, "validation")
    meta = {
        "channels": list(cfg.channels),
        "T": cfg.T,
        "preprocessing_version": cfg.preprocessing_version,
        "config_hash": chash,
        "fold": fold,
        "split_manifest_hash": split_hash,
        "protocol": "P1_LOYO",
        "best_epoch": best_epoch,
        "val_auprc": best_auprc,
        "seed": cfg.seed,
        "is_mock": bool(source.is_mock),
        "loss_impl": LOSS_IMPL,
    }
    save_checkpoint(model.cpu(), paths["ckpt"], meta, strict=True)
    selection = {
        "protocol": "P1",
        "fold": fold,
        "fitted_on": "validation",
        "threshold": asdict(threshold),
        "calibrator": asdict(calibrator),
        "calibration_note": "Platt scaling of model probability; uncertainty layer NOT produced",
    }
    paths["sel"].write_text(json.dumps(selection, indent=2) + "\n")
    summary = {
        "run": run,
        "fold": fold,
        "is_mock": bool(source.is_mock),
        "config": asdict(cfg),
        "config_hash": chash,
        "split_manifest_hash": split_hash,
        "checkpoint": paths["ckpt"].name,
        "checkpoint_sha256": hashlib.sha256(paths["ckpt"].read_bytes()).hexdigest(),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "val_auprc": best_auprc,
        "val_threshold": threshold.threshold,
        "train_samples": len(train_keys),
        "val_samples": len(val_keys),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.time() - t0, 1),
        "hardware": hw.as_dict(),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "loss_impl": LOSS_IMPL,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str) + "\n")
    for k in ("last", "best"):  # only the best model is kept once the fold is finished
        paths[k].unlink(missing_ok=True)
    log(f"[{run} fold {fold}] finished: best epoch {best_epoch}, validation AUPRC "
        f"{best_auprc:.4f}, {summary['seconds'] / 60:.1f} min")
    return summary


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
