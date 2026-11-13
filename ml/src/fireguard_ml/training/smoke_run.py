"""Synthetic plumbing test configured from the canonical unet.yaml."""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch

from fireguard_ml.data.channel_schema import PROCESSED_WSTS_CHANNELS
from fireguard_ml.data.sample_adapter import adapt_sample
from fireguard_ml.models import UNet
from fireguard_ml.training.config import configure_reproducibility, load_config
from fireguard_ml.training.losses import masked_bce_dice_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/unet.yaml")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--height", type=int, default=304)
    parser.add_argument("--width", type=int, default=247)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    config, config_sha = load_config(args.config)
    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"]["deterministic"])
    configure_reproducibility(seed, deterministic)
    if args.channels != len(PROCESSED_WSTS_CHANNELS):
        raise ValueError("Synthetic smoke channels must match the documented processed schema")
    channel_names = PROCESSED_WSTS_CHANNELS
    x = torch.randn(1, args.channels, args.height, args.width)
    y = torch.zeros(1, args.height, args.width)
    y[..., args.height // 2, args.width // 2] = 1
    sample = adapt_sample(
        x,
        y,
        channel_names=channel_names,
        source_input_valid_mask=torch.ones(1, 1, args.height, args.width, dtype=torch.bool),
        source_target_valid_mask=torch.ones(1, 1, args.height, args.width, dtype=torch.bool),
        pad_multiple=int(config["spatial"]["pad_multiple"]),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNet(
        sample.inputs.shape[1],
        base_channels=args.base_channels or int(config["model"]["base_channels"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = sample.inputs.to(device)
    target = sample.target.to(device)
    valid = sample.loss_valid_mask.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = masked_bce_dice_loss(
            logits,
            target,
            valid,
            bce_weight=float(config["loss"]["bce_weight"]),
            dice_weight=float(config["loss"]["dice_weight"]),
        )
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "kind": "SYNTHETIC_SMOKE_ONLY_NOT_WSTS_BENCHMARK",
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "input_shape": list(inputs.shape),
                "output_shape": list(logits.shape),
                "validity_status": sample.validity_status,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "steps": args.steps,
                "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / args.steps,
                "last_loss": float(loss.detach().cpu()),
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "config_sha256": config_sha,
                "seed": seed,
                "deterministic_algorithms_requested": deterministic,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
