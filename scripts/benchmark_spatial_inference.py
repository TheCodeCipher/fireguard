"""Synthetic Role 2 spatial-inference robustness/efficiency probe (not M1 evidence)."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import torch

from fireguard_ml.data.padding import unpad
from fireguard_ml.data.sample_adapter import adapt_inputs_for_inference
from fireguard_ml.models.checkpointing import save_spatial_checkpoint
from fireguard_ml.models.unet import UNet
from fireguard_ml.training.config import configure_reproducibility


def _system_ram_bytes() -> int | None:
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
    if get_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return None


def _parse_sizes(values: list[str]) -> list[tuple[int, int]]:
    sizes = []
    for value in values:
        try:
            height, width = (int(part) for part in value.lower().split("x", maxsplit=1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Expected HEIGHTxWIDTH, got {value!r}") from exc
        if min(height, width) < 1:
            raise argparse.ArgumentTypeError("Frame dimensions must be positive")
        sizes.append((height, width))
    return sizes


def run(args: argparse.Namespace) -> dict:
    if args.channels < 1 or args.base_channels < 1 or args.measured_steps < 1:
        raise ValueError("channels, base_channels, and measured_steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    configure_reproducibility(args.seed, deterministic=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    names = tuple(f"synthetic_channel_{i}" for i in range(args.channels))
    model = UNet(args.channels, base_channels=args.base_channels).to(device).eval()
    shapes = []
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for height, width in _parse_sizes(args.sizes):
        x = torch.randn(1, args.channels, height, width, generator=generator)
        valid = torch.ones(1, 1, height, width, dtype=torch.bool)
        sample = adapt_inputs_for_inference(
            x,
            channel_names=names,
            source_input_valid_mask=valid,
            pad_multiple=32,
        )
        padded = sample.inputs.to(device)

        def forward() -> torch.Tensor:
            with torch.inference_mode():
                return model(padded)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(args.warmup_steps):
            logits = forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(args.measured_steps):
            start = time.perf_counter()
            logits = forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
        restored = unpad(logits, sample.pad_info)
        if restored.shape[-2:] != (height, width):
            raise RuntimeError("Unpadded output does not match input frame size")
        shapes.append(
            {
                "original_hw": [height, width],
                "padded_hw": [sample.pad_info.padded_height, sample.pad_info.padded_width],
                "output_hw_after_unpad": list(restored.shape[-2:]),
                "valid_pixels": int(sample.input_valid_mask.sum()),
                "mean_inference_seconds": statistics.mean(timings),
                "median_inference_seconds": statistics.median(timings),
                "frames_per_second_batch1": 1.0 / statistics.mean(timings),
            }
        )

    meta = {
        "model_version": "synthetic-spatial-probe-only",
        "channel_names": list(names),
        "data_contract_version": "synthetic-only",
        "config_sha256": "synthetic-only",
        "split_manifest_sha256": "synthetic-only",
        "test_fold_year": 2018,
        "seed": args.seed,
        "environment": {"kind": "synthetic benchmark", "device": str(device)},
        "training_group_ids": [],
        "validation_group_ids": [],
        "strict": False,
    }
    with tempfile.TemporaryDirectory(prefix="fireguard-spatial-probe-") as temp_dir:
        checkpoint = Path(temp_dir) / "synthetic-spatial.pt"
        save_spatial_checkpoint(model.cpu(), checkpoint, meta)
        checkpoint_bytes = checkpoint.stat().st_size

    report = {
        "kind": "SYNTHETIC_SPATIAL_EFFICIENCY_PROBE_NOT_M1_RESULT",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "system_ram_bytes": _system_ram_bytes(),
        "process_peak_working_set_bytes": _process_peak_working_set_bytes(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "batch_size": 1,
        "channels": args.channels,
        "channel_schema_status": "SYNTHETIC_ONLY_NOT_ROLE4_APPROVED",
        "base_channels": args.base_channels,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "synthetic_checkpoint_size_bytes": checkpoint_bytes,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "full_frame_cases": shapes,
        "limitations": [
            "Synthetic inputs and an untrained U-Net only; no WSTS performance or accuracy claim.",
            "Does not satisfy the representative real-data/Gate 1 benchmark requirement.",
            "No training, split evaluation, or Phase 2 scale-up was performed.",
        ],
    }
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["304x247", "305x207"])
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--device")
    parser.add_argument("--report")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
