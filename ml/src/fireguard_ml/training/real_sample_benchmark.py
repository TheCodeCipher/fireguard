"""Short Role 2 real-sample compute probe (not M1 training/evaluation)."""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import os
import platform
import statistics
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from fireguard_ml.data.channel_schema import PROCESSED_WSTS_CHANNELS
from fireguard_ml.data.sample_adapter import adapt_sample, restore_frame
from fireguard_ml.models import UNet
from fireguard_ml.training.config import configure_reproducibility, load_config
from fireguard_ml.training.losses import masked_bce_dice_loss


def _system_memory_bytes() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullTotalPhys)
    return None


def _process_peak_working_set_bytes() -> int | None:
    """Return the process high-water working set on Windows, when available."""
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong)
    get_info.restype = ctypes.c_int
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = get_info(process, ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def _load_export(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if not {"x.npy", "y.npy"}.issubset(names):
            raise ValueError(f"Expected x.npy/y.npy, found {sorted(names)}")
        x = np.load(io.BytesIO(archive.read("x.npy")), allow_pickle=False)
        y = np.load(io.BytesIO(archive.read("y.npy")), allow_pickle=False)
    return x, y, time.perf_counter() - start


def _load_mask(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    return torch.from_numpy(np.load(path, allow_pickle=False).copy()).bool()


def run(args: argparse.Namespace) -> dict:
    config, config_hash = load_config(args.config)
    seed = int(config["reproducibility"].get("seed", 2026))
    deterministic = bool(config["reproducibility"].get("deterministic", True))
    configure_reproducibility(seed, deterministic)

    x_np, y_np, data_load_seconds = _load_export(Path(args.sample_zip))
    if x_np.dtype != np.float32 or y_np.dtype != np.int64:
        raise ValueError(f"Unexpected sample dtypes: x={x_np.dtype}, y={y_np.dtype}")
    if x_np.ndim != 4 or x_np.shape[0] != 1:
        raise ValueError("Spatial benchmark export input must be [1,C,H,W]")

    selected_names = config["channel_schema"].get("selected_names")
    strict = bool(config["channel_schema"].get("strict_approved", False))
    approval_reference = config["channel_schema"].get("approval_reference")
    if selected_names is None:
        selected_names = PROCESSED_WSTS_CHANNELS
    selected_names = tuple(selected_names)
    input_valid = _load_mask(args.input_valid_mask)
    target_valid = _load_mask(args.target_valid_mask)
    masks_missing = input_valid is None or target_valid is None
    if masks_missing and not args.allow_provisional_finite_mask:
        raise ValueError(
            "Authoritative input and target masks are required. For a provisional probe only, "
            "pass --allow-provisional-finite-mask."
        )
    adapt_start = time.perf_counter()
    sample = adapt_sample(
        torch.from_numpy(x_np.copy()),
        torch.from_numpy(y_np.copy()),
        selected_channels=selected_names,
        strict_approved=strict,
        approval_reference=approval_reference,
        source_input_valid_mask=input_valid,
        source_target_valid_mask=target_valid,
        allow_finite_mask_fallback=args.allow_provisional_finite_mask,
        target_nodata_value=config["data"].get("target_nodata_value"),
        pad_multiple=int(config["spatial"]["pad_multiple"]),
        event_id=args.event_id,
        origin=args.input_date,
        target_day=args.target_date,
    )
    adapt_seconds = time.perf_counter() - adapt_start

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_reproducibility(seed, deterministic)
    base_channels = int(config["model"]["base_channels"])
    model = UNet(sample.inputs.shape[1], base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    inputs, target, valid = (
        sample.inputs.to(device),
        sample.target.to(device),
        sample.loss_valid_mask.to(device),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    model.train()

    def optimizer_step() -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        step_logits = model(inputs)
        step_loss = masked_bce_dice_loss(
            step_logits,
            target,
            valid,
            bce_weight=float(config["loss"]["bce_weight"]),
            dice_weight=float(config["loss"]["dice_weight"]),
        )
        step_loss.backward()
        optimizer.step()
        return step_logits, step_loss

    for _ in range(args.warmup_steps):
        optimizer_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    measured_times = []
    for _ in range(args.measured_steps):
        start = time.perf_counter()
        logits, loss = optimizer_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measured_times.append(time.perf_counter() - start)
    compute_seconds = statistics.mean(measured_times)
    restored = restore_frame(logits.detach(), sample)

    gpu_properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    report = {
        "kind": "SHORT_REAL_SAMPLE_COMPUTE_PROBE_NOT_M1_TRAINING_OR_EVALUATION",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "sample": {
            "event_id": args.event_id,
            "input_date": args.input_date,
            "target_date": args.target_date,
            "input_shape_source": list(x_np.shape),
            "target_shape_source": list(y_np.shape),
            "target_positive_pixels": int((y_np > 0).sum()),
            "source_aware_masks_supplied": not masks_missing,
            "validity_status": sample.validity_status,
            "input_valid_pixels": int(sample.input_valid_mask.sum()),
            "target_valid_pixels": int(sample.target_valid_mask.sum()),
            "loss_valid_pixels": int(sample.loss_valid_mask.sum()),
            "export_frame_is_full_original_frame": args.export_is_full_frame,
        },
        "model": {
            "selected_channels": list(sample.channel_selection.names),
            "channel_selection_status": sample.channel_selection.selection_status,
            "channel_schema_sha256": sample.channel_selection.sha256,
            "strict_approved": strict,
            "approval_reference": approval_reference,
            "input_shape_after_adapter": list(sample.inputs.shape),
            "padded_hw": [sample.pad_info.padded_height, sample.pad_info.padded_width],
            "logits_shape": list(logits.shape),
            "restored_logits_shape": list(restored.shape),
            "base_channels": base_channels,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "timing_seconds": {
            "export_load": data_load_seconds,
            "adapt_and_pad": adapt_seconds,
            "mean_forward_loss_backward_optimizer_step": compute_seconds,
            "median_forward_loss_backward_optimizer_step": statistics.median(measured_times),
            "steps_per_second": 1.0 / compute_seconds,
        },
        "compute": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "os": platform.platform(),
            "cpu": platform.processor(),
            "system_ram_bytes": _system_memory_bytes(),
            "process_peak_working_set_bytes": _process_peak_working_set_bytes(),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_memory_bytes": gpu_properties.total_memory if gpu_properties else None,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "batch_size": int(inputs.shape[0]),
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "seed": seed,
            "deterministic_algorithms_requested": deterministic,
        },
        "loss": float(loss.detach().cpu()),
        "config_sha256": config_hash,
        "split_manifest_path": config["data"].get("split_manifest_path"),
        "split_manifest_sha256": config["data"].get("split_manifest_sha256"),
        "limitations": [
            (
                "One sample and a short optimizer-step probe are compute/interface evidence only, "
                "not a trained baseline or predictive-quality result."
            ),
            (
                "When source masks are absent, the finite-only mask is provisional and "
                "must not be used for official real-data training/evaluation."
            ),
            (
                "The export is 288x224; Role 1's notes report a 304x247 raster center-crop. "
                "This export cannot restore discarded borders, so full-frame agreement is pending."
            ),
            (
                "All 40 channels remain exploratory until the config carries Role 4's "
                "reviewed strict subset and approval reference."
            ),
        ],
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_zip")
    parser.add_argument("--config", default="../configs/unet.yaml")
    parser.add_argument("--input-valid-mask")
    parser.add_argument("--target-valid-mask")
    parser.add_argument("--allow-provisional-finite-mask", action="store_true")
    parser.add_argument("--export-is-full-frame", action="store_true")
    parser.add_argument("--event-id", default="fire_21458798")
    parser.add_argument("--input-date", default="2018-01-06")
    parser.add_argument("--target-date", default="2018-01-07")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--device")
    parser.add_argument("--report")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
