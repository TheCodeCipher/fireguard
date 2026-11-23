"""Pick device, precision, batch size and data-loading workers for the machine at hand.

The aim is to keep the GPU busy without running out of memory:

* device: CUDA if present, else Apple MPS, else CPU;
* precision: bfloat16 autocast on GPUs that support it (no loss scaling needed), float16 with a
  gradient scaler on older CUDA GPUs, float32 on CPU;
* batch size: measured, not guessed. One real training step (forward, backward, optimizer) is
  run at the largest padded frame size for growing batch sizes; the largest that stays under
  ``mem_fraction`` of the GPU memory is kept (0.65: PyTorch's memory cache grows past the probed
  peak over hours of training, and on Windows the overflow silently spills into system RAM);
* gradient accumulation tops the batch up to the configured effective batch, so the optimiser
  sees the same number of samples per update on every machine (results do not depend on the GPU);
* workers: CPU cores minus two (one for the main process, one for the OS), at most 8.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Hardware:
    device: str
    amp_dtype: str | None  # "bfloat16" | "float16" | None
    batch_size: int
    grad_accum: int
    num_workers: int
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    probed_peak_gb: float | None = None

    @property
    def torch_amp_dtype(self):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(self.amp_dtype or "")

    def as_dict(self) -> dict:
        return asdict(self)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_workers(can_use_processes: bool) -> int:
    if not can_use_processes:
        return 0
    return max(0, min(8, (os.cpu_count() or 2) - 2))


def _probe_step(model_factory, n_ch: int, T: int, hw: tuple[int, int], bs: int,
                device: torch.device, amp_dtype) -> int:
    """Run one full training step; return peak allocated bytes."""
    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(bs, T, n_ch, *hw, device=device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        loss = model(x).float().mean()
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated(device)
    del model, opt, x, loss
    torch.cuda.empty_cache()
    return peak


def plan_hardware(model_factory, n_ch: int, T: int, max_frame_hw: tuple[int, int],
                  effective_batch: int, device: str | None = None, amp: bool = True,
                  max_batch: int = 16, mem_fraction: float = 0.65, batch_size: int | None = None,
                  num_workers: int | None = None, can_use_processes: bool = True,
                  log=print) -> Hardware:
    dev = pick_device(device)
    workers = default_workers(can_use_processes) if num_workers is None else num_workers
    if dev.type != "cuda":
        bs = batch_size or (4 if dev.type == "mps" else 2)
        bs = min(bs, effective_batch)
        hw = Hardware(str(dev), None, bs, math.ceil(effective_batch / bs), workers)
        log(f"[hardware] {dev.type}: batch {bs} x accumulation {hw.grad_accum}, {workers} workers")
        return hw
    amp_dtype = None
    if amp:
        # Native bfloat16 needs compute capability 8.0+ (Ampere: RTX 30xx, A100, ...). Newer
        # PyTorch reports bf16 as "supported" on older GPUs (T4, V100) through slow emulation, so
        # the capability is checked directly; those GPUs use float16 with loss scaling instead.
        native_bf16 = torch.cuda.get_device_capability(dev)[0] >= 8
        amp_dtype = "bfloat16" if native_bf16 else "float16"
    tdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype or "")
    props = torch.cuda.get_device_properties(dev)
    total = props.total_memory
    hw_pad = tuple(-(-s // 32) * 32 for s in max_frame_hw)
    best, peak_best = 1, None
    if batch_size is not None:
        best = batch_size
    else:
        bs = 1
        while bs <= min(max_batch, effective_batch):
            try:
                peak = _probe_step(model_factory, n_ch, T, hw_pad, bs, dev, tdtype)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
            if peak > mem_fraction * total:
                break
            best, peak_best = bs, peak
            bs *= 2
    hw = Hardware(str(dev), amp_dtype, best, math.ceil(effective_batch / best), workers,
                  props.name, round(total / 2**30, 2),
                  round(peak_best / 2**30, 2) if peak_best else None)
    log(f"[hardware] {props.name} ({hw.gpu_memory_gb} GB), {amp_dtype or 'float32'}: batch "
        f"{best} x accumulation {hw.grad_accum} (probe peak {hw.probed_peak_gb} GB at "
        f"{hw_pad[0]}x{hw_pad[1]}), {workers} loader workers")
    return hw
