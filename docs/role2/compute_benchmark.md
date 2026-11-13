# Compute benchmark — Role 2

## Synthetic plumbing run (completed)

This is a **synthetic smoke benchmark only**, not a WSTS training result. Random values were generated at the documented original sample dimensions; no WSTS raster/tensor was loaded.

- Python: 3.13.3 (project guide target remains Python 3.11)
- PyTorch: 2.15.0+cpu
- Device: CPU; CUDA unavailable in this environment
- GPU/fallback: CPU smoke route verified; a separate agreed GPU/Colab/university route is still needed for representative Phase 1 compute planning
- Input: `[1, 40, 304, 247]` synthetic, using the documented processed WSTS channel schema
- Padding: bottom/right from `304×247` to `320×256`; all `75,088` original pixels valid in this synthetic sample
- Model: U-Net, base width 4, 31,873 parameters
- Batch size: 1
- Optimizer: Adam, learning rate `1e-3`
- Steps: 2
- Elapsed training time: 0.2259 seconds total (0.1130 seconds/step)
- Peak GPU memory: not applicable (CPU)
- Final synthetic loss: 1.5963
- Test command: `python -m fireguard_ml.training.smoke_run --steps 2 --height 304 --width 247 --channels 40 --base-channels 4`
- Shape/mask test result: 9 passed

This run confirms only that the scaffold can forward/backpropagate at a representative padded shape on CPU. It is not representative of real preprocessing, data loading, WSTS optimization, or project training throughput. Repeat the prescribed short benchmark with Role 1's real sample and the project's agreed Python 3.11 compute route when available.

## Remaining benchmark fields

- Real sample loader time:
- Approved strict channel schema/hash (Role 4):
- Frozen split manifest/hash (Role 4):
- Data-loading timing:
- Representative batch size / steps:
- Agreed GPU/fallback model and VRAM:
- Phase 2 runtime estimate:
