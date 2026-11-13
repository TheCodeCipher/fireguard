"""Measure peak GPU memory and step time for M3/M4 at the largest WSTS frame (Role 3).

    python scripts/benchmark_temporal_memory.py --T 5 3

Uses random tensors of the padded maximum frame (356x311 -> 384x320), the strict channel count
and batch 1 with AMP: the real training step shape. Needs no WSTS data, so it can run now and
tells you whether M4 at T=5 fits a 6 GB card before any training is scheduled.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from fireguard_ml.models.temporal_unet import MODEL_REGISTRY

MAX_HW = (356, 311)  # largest frame in the WSTS metadata


def pad32(n: int) -> int:
    return -(-n // 32) * 32


def bench(model_type: str, T: int, c: int, base: int, steps: int, device) -> dict:
    kw = dict(c_strict=c, T=T, base=base, depth=5)
    if model_type == "m4_convlstm_unet":
        kw["lstm_kernel"] = 3
    model = MODEL_REGISTRY[model_type](**kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    H, W = pad32(MAX_HW[0]), pad32(MAX_HW[1])
    x = torch.randn(1, T, c, H, W, device=device)
    y = (torch.rand(1, 1, H, W, device=device) > 0.99).float()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    try:
        for _ in range(steps):
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        status = "ok"
    except torch.OutOfMemoryError:
        status = "OOM"
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "model": model_type,
        "T": T,
        "channels": c,
        "base": base,
        "padded_hw": [H, W],
        "status": status,
        "peak_gpu_gib": round(peak / 2**30, 2) if peak else None,
        "median_step_s": round(sorted(times)[len(times) // 2], 3) if times else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--T", type=int, nargs="+", default=[5, 3])
    ap.add_argument("--channels", type=int, default=7)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in ("m3_stacked_unet", "m4_convlstm_unet"):
        for T in a.T:
            print(json.dumps(bench(m, T, a.channels, a.base, a.steps, device)))
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
