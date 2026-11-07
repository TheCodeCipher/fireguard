"""Reproducible synthetic shape/loss/timing demo; not a WSTS benchmark."""
from __future__ import annotations

import argparse
import json
import platform
import time

import torch

from fireguard_ml.data.sample_adapter import adapt_sample
from fireguard_ml.models import UNet
from fireguard_ml.training.losses import masked_bce_dice_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--base-channels", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Synthetic analogue of documented WSTS tensor dimensions; values are random.
    x = torch.randn(1, args.channels, args.height, args.width)
    y = torch.zeros(1, args.height, args.width)
    y[..., args.height // 2, args.width // 2] = 1
    sample = adapt_sample(x, y)
    model = UNet(sample.inputs.shape[1], base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb, yb, mb = sample.inputs.to(device), sample.target.to(device), sample.valid_mask.to(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    last_loss = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = masked_bce_dice_loss(logits, yb, mb)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    report = {
        "kind": "SYNTHETIC_SMOKE_ONLY_NOT_WSTS_BENCHMARK",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_shape": list(xb.shape),
        "target_shape": list(yb.shape),
        "valid_pixels": int(mb.sum().item()),
        "output_shape": list(logits.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "last_loss": last_loss,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
