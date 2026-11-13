# FireGuard Role 2 — Spatial Model Starter

This folder now contains an executable U-Net scaffold using the documented processed WSTS tensor interface. It is structured so later data/channel/split handoffs can be plugged in without rewriting the model.

## What is implemented

- Compact 2D U-Net returning per-pixel raw logits.
- Named schema for the documented 40-channel processed WSTS baseline example.
- Named-channel selection adapter; channel order/subset can be replaced in config after Role 4's strict no-look-ahead decision.
- Sample adapter accepts `[B,C,H,W]` plus target, optional source validity mask, event/date metadata, pads bottom/right to multiples of 32, and preserves `PadInfo` for exact unpadding.
- Conservative finite-input validity mask. Replace with Role 1's source-aware mask when handed off.
- Masked BCE + soft Dice loss.
- Synthetic shape/mask checks and a synthetic timing command.

The 40 channel names and the example shape are taken from the Role 1 data dictionary/reproduction handoff. This is the *processed baseline schema*, not approval to use all channels in strict M1. The config marks it exploratory and keeps `strict_approved: false`. Do not treat synthetic benchmark numbers as WSTS model performance.

## Run the included checks

Project guide target: Python 3.11. Install the team's pinned PyTorch and pytest versions, then from this folder run:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python -m fireguard_ml.training.smoke_run --steps 2 --height 304 --width 247 --channels 40 --base-channels 4
```

The smoke command uses synthetic random values, on CPU or CUDA if available. It prints an explicit `SYNTHETIC_SMOKE_ONLY_NOT_WSTS_BENCHMARK` label.

## Easy swap when handoffs arrive

1. Replace `channel_schema.selected_names` with Role 4's approved channel names in approved order. Keep `strict_approved: false` until that review is recorded; then set true and use the explicit subset.
2. Replace the synthetic sample creation with Role 1's loader, then call `adapt_sample(inputs, target, source_valid_mask=..., event_id=..., origin=..., target_day=...)`.
3. Read Role 4's frozen split manifest in the future training entry point; its path/hash belongs in run metadata/config. The model has no hard-coded split logic.
4. Keep the adapter and model tensor contract stable: `[B,C,H,W]` in; `[B,1,H_pad,W_pad]` logits out; unpad before full-frame evaluation.

## Scope limits

- The geospatial ZIP contains metadata and manifests but no actual WSTS tensor or TIFF payload, so real-data training is not performed here.
- Strict channel approval and the frozen split manifest remain pending.
- Current timing used Python 3.13 CPU; the controlling plan targets Python 3.11 and requires a representative agreed GPU/fallback measurement for Gate 1.
- Geometric augmentation stays off. Keep full frames; pad and mask, then unpad for evaluation.
