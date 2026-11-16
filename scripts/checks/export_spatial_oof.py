"""Command-line entry point for target-blind spatial OOF export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireguard_ml.inference.spatial import run_spatial_oof_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one held-out spatial probability map from registered fold metadata. "
            "This command reads input tensors and masks only; it does not load targets."
        )
    )
    parser.add_argument("--event-id", required=True, help="Event ID from the P1 manifest")
    parser.add_argument("--sample-id", required=True, help="Input sample/origin ID")
    parser.add_argument("--input-archive", required=True, type=Path, help="NPZ or ZIP with x and input_valid_mask")
    parser.add_argument("--release-manifest", required=True, type=Path, help="Local release JSON for fold checkpoints")
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Root used to resolve relative manifest/checkpoint paths",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination compressed probability NPZ")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = run_spatial_oof_inference(
        event_id=args.event_id,
        sample_id=args.sample_id,
        input_archive=args.input_archive,
        release_manifest_path=args.release_manifest,
        project_root=args.project_root,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
