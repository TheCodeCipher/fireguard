"""M3 (stacked-days U-Net) and M4 (shared encoder + ConvLSTM U-Net) for FireGuard (Role 3).

Shared contract (docs/temporal_input_contract.md)
    input  x : [B, T, C, H_pad, W_pad]   C = number of STRICT (no-GFS) channels, frames oldest -> newest
    output   : [B, 1, H_pad, W_pad] logits (one per cell)
    H_pad and W_pad must be multiples of 2**depth (default depth=5 -> 32).

Loss, valid-mask handling and metrics are NOT in here (Roles 2 and 4). `predict()` is the offline
inference path: sigmoid, optional crop of the bottom/right padding. Backend never calls it.

Design decision to agree with Role 2: M4 takes skip connections from the LAST frame (day D) and applies
the ConvLSTM at the bottleneck. M3 and M4 use the same encoder/decoder blocks for a fair comparison.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .convlstm import ConvLSTM


def _norm(ch: int) -> nn.GroupNorm:
    # GroupNorm: no dependence on batch statistics (batch size is small on full frames)
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch), nn.ReLU(inplace=True),
        )


def level_channels(base: int, depth: int) -> List[int]:
    return [base * 2 ** min(i, 3) for i in range(depth + 1)]


class Encoder(nn.Module):
    def __init__(self, in_ch: int, base: int, depth: int):
        super().__init__()
        ch = level_channels(base, depth)
        self.ch = ch
        self.stem = ConvBlock(in_ch, ch[0])
        self.downs = nn.ModuleList(ConvBlock(ch[i - 1], ch[i]) for i in range(1, depth + 1))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(x)]
        for blk in self.downs:
            feats.append(blk(self.pool(feats[-1])))
        return feats  # feats[-1] is the bottleneck at 1/2**depth resolution


class Decoder(nn.Module):
    def __init__(self, base: int, depth: int):
        super().__init__()
        ch = level_channels(base, depth)
        self.ups = nn.ModuleList(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth, 0, -1))
        self.blocks = nn.ModuleList(ConvBlock(2 * ch[i - 1], ch[i - 1]) for i in range(depth, 0, -1))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = bottleneck
        for up, blk, skip in zip(self.ups, self.blocks, reversed(skips[:-1])):
            x = blk(torch.cat([up(x), skip], dim=1))
        return self.head(x)


class _TemporalBase(nn.Module):
    model_type = "base"

    def __init__(self, c_strict: int, T: int, base: int, depth: int):
        super().__init__()
        if c_strict < 1 or T < 1:
            raise ValueError("c_strict and T must be >= 1")
        self.c_strict, self.T, self.base, self.depth = c_strict, T, base, depth

    def spec(self) -> Dict[str, object]:
        return dict(model_type=self.model_type, c_strict=self.c_strict, T=self.T, base=self.base, depth=self.depth)

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(x.shape)}")
        _, t, c, h, w = x.shape
        if t != self.T or c != self.c_strict:
            raise ValueError(f"expected T={self.T}, C={self.c_strict}; got T={t}, C={c}")
        m = 2 ** self.depth
        if h % m or w % m:
            raise ValueError(f"H and W must be multiples of {m} (pad first); got {h}x{w}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor, orig_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Offline inference: probabilities [B, 1, H, W]. Crops bottom/right padding if orig_hw is given."""
        was_training = self.training
        self.eval()
        try:
            p = torch.sigmoid(self(x))
        finally:
            self.train(was_training)
        if orig_hw is not None:
            p = p[..., : orig_hw[0], : orig_hw[1]]
        return p


class StackedDaysUNet(_TemporalBase):
    """M3: stack T frames along channels -> [B, T*C, H, W] -> U-Net."""

    model_type = "m3_stacked_unet"

    def __init__(self, c_strict: int, T: int = 5, base: int = 32, depth: int = 5):
        super().__init__(c_strict, T, base, depth)
        self.enc = Encoder(T * c_strict, base, depth)
        self.dec = Decoder(base, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        b, t, c, h, w = x.shape
        feats = self.enc(x.reshape(b, t * c, h, w))
        return self.dec(feats[-1], feats)


class ConvLSTMUNet(_TemporalBase):
    """M4: shared per-frame encoder -> ConvLSTM over time at the bottleneck -> decoder (skips from last frame)."""

    model_type = "m4_convlstm_unet"

    def __init__(self, c_strict: int, T: int = 5, base: int = 32, depth: int = 5, lstm_kernel: int = 3):
        super().__init__(c_strict, T, base, depth)
        self.enc = Encoder(c_strict, base, depth)
        bott = self.enc.ch[-1]
        self.temporal = ConvLSTM(bott, bott, kernel_size=lstm_kernel)
        self.dec = Decoder(base, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        b, t, c, h, w = x.shape
        feats = self.enc(x.reshape(b * t, c, h, w))  # same weights for every day
        bott = feats[-1].reshape(b, t, *feats[-1].shape[1:])
        _, (h_last, _) = self.temporal(bott)
        last_skips = [f.reshape(b, t, *f.shape[1:])[:, -1] for f in feats[:-1]]
        return self.dec(h_last, last_skips + [h_last])


MODEL_REGISTRY = {StackedDaysUNet.model_type: StackedDaysUNet, ConvLSTMUNet.model_type: ConvLSTMUNet}
