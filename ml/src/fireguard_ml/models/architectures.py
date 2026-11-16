"""The four FireGuard forecast models (M1-M4). Each maps recent days to a next-day fire map.

Shared contract (docs/contracts/temporal_input_contract.md)
    input  x : [B, T, C, H_pad, W_pad]   C = strict channels, frames oldest -> newest
    output   : [B, 1, H_pad, W_pad] logits, one per 375 m cell (sigmoid -> probability)
    H_pad and W_pad must be multiples of 2**depth (depth 5 -> 32).

The ladder isolates one design change per step, so each comparison answers one question:

    M1 spatial U-Net         day D only (T=1), plain conv blocks          reference CNN
    M2 attention res-U-Net   day D only (T=1), residual blocks + attention gates
                             question: does a stronger spatial architecture help (M2 vs M1)?
    M3 stacked-days U-Net    T days stacked on channels, M1's blocks
                             question: does temporal context help (M3 vs M1)?
    M4 ConvLSTM U-Net        shared per-day encoder, ConvLSTM memory at the two coarsest levels
                             question: does recurrence beat stacking (M4 vs M3)?

Why M2 is not "U-Net + terrain + land cover" (plan v4 5.2): land cover is year-Y MODIS and fails
the no-look-ahead audit (docs/data/leakage_audit.csv), and terrain is already in the strict channels
every model sees. The terrain question is answered by the ``m1_no_terrain`` ablation instead.

Regularisation (applies to all): GroupNorm (independent of batch size, which is small on full
frames) and spatial dropout at the bottleneck (``dropout``, default 0.1 in configs/models.yaml).
Loss, valid masks and metrics live elsewhere (training/, evaluation/).
"""

from __future__ import annotations

import torch
from torch import nn

from .convlstm import ConvLSTM


def _norm(ch: int) -> nn.GroupNorm:
    # GroupNorm: no dependence on batch statistics (batch size is small on full frames)
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class ConvBlock(nn.Sequential):
    """Two 3x3 conv -> GroupNorm -> ReLU layers (the classic U-Net block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Residual version of ConvBlock: the input is added back (1x1 projection if widths differ),
    which keeps gradients flowing through deep encoders (He et al., 2016)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch),
        )
        self.skip = (nn.Identity() if in_ch == out_ch else
                     nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), _norm(out_ch)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class AttentionGate(nn.Module):
    """Additive attention gate on a skip connection (Oktay et al., 2018, Attention U-Net).

    The decoder signal ``g`` decides which skip cells matter; the skip map is multiplied by a
    per-cell weight in [0, 1]. Useful here because fire occupies a tiny fraction of each frame.
    """

    def __init__(self, g_ch: int, x_ch: int):
        super().__init__()
        inter = max(x_ch // 2, 8)
        self.wg = nn.Sequential(nn.Conv2d(g_ch, inter, 1, bias=False), _norm(inter))
        self.wx = nn.Sequential(nn.Conv2d(x_ch, inter, 1, bias=False), _norm(inter))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter, 1, 1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.wg(g) + self.wx(x))


def level_channels(base: int, depth: int) -> list[int]:
    return [base * 2 ** min(i, 3) for i in range(depth + 1)]


class Encoder(nn.Module):
    def __init__(self, in_ch: int, base: int, depth: int, block=ConvBlock):
        super().__init__()
        ch = level_channels(base, depth)
        self.ch = ch
        self.stem = block(in_ch, ch[0])
        self.downs = nn.ModuleList(block(ch[i - 1], ch[i]) for i in range(1, depth + 1))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = [self.stem(x)]
        for blk in self.downs:
            feats.append(blk(self.pool(feats[-1])))
        return feats  # feats[-1] is the bottleneck at 1/2**depth resolution


class Decoder(nn.Module):
    def __init__(self, base: int, depth: int, block=ConvBlock, attention: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        ch = level_channels(base, depth)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth, 0, -1))
        self.gates = nn.ModuleList(
            AttentionGate(ch[i - 1], ch[i - 1]) if attention else nn.Identity()
            for i in range(depth, 0, -1))
        self.attention = attention
        self.blocks = nn.ModuleList(
            block(2 * ch[i - 1], ch[i - 1]) for i in range(depth, 0, -1))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """skips holds exactly `depth` encoder maps, shallowest first; bottleneck is separate."""
        x = self.drop(bottleneck)
        for up, gate, blk, skip in zip(self.ups, self.gates, self.blocks, reversed(skips),
                                       strict=True):
            x = up(x)
            if self.attention:
                skip = gate(x, skip)
            x = blk(torch.cat([x, skip], dim=1))
        return self.head(x)


class _TemporalBase(nn.Module):
    model_type = "base"

    def __init__(self, c_strict: int, T: int, base: int, depth: int, dropout: float = 0.0):
        super().__init__()
        if c_strict < 1 or T < 1:
            raise ValueError("c_strict and T must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.c_strict, self.T, self.base, self.depth = c_strict, T, base, depth
        self.dropout = dropout

    def spec(self) -> dict[str, object]:
        return dict(model_type=self.model_type, c_strict=self.c_strict, T=self.T,
                    base=self.base, depth=self.depth, dropout=self.dropout)

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
    def predict(self, x: torch.Tensor, orig_hw: tuple[int, int] | None = None) -> torch.Tensor:
        """Offline inference: probabilities [B, 1, H, W]. Crops padding if orig_hw is given."""
        was_training = self.training
        self.eval()
        try:
            p = torch.sigmoid(self(x).float())
        finally:
            self.train(was_training)
        if orig_hw is not None:
            p = p[..., : orig_hw[0], : orig_hw[1]]
        return p


class StackedDaysUNet(_TemporalBase):
    """M3: stack T frames along channels -> [B, T*C, H, W] -> U-Net (early temporal fusion)."""

    model_type = "m3_stacked_unet"

    def __init__(self, c_strict: int, T: int = 5, base: int = 32, depth: int = 5,
                 dropout: float = 0.0):
        super().__init__(c_strict, T, base, depth, dropout)
        self.enc = Encoder(T * c_strict, base, depth)
        self.dec = Decoder(base, depth, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        b, t, c, h, w = x.shape
        feats = self.enc(x.reshape(b, t * c, h, w))
        return self.dec(feats[-1], feats[:-1])


class SpatialUNet(StackedDaysUNet):
    """M1: the reference spatial U-Net on day D only (T = 1).

    Same encoder/decoder as M3, so M1 vs M3 changes only the number of input days.
    """

    model_type = "m1_spatial_unet"

    def __init__(self, c_strict: int, T: int = 1, base: int = 32, depth: int = 5,
                 dropout: float = 0.0):
        if T != 1:
            raise ValueError("M1 is spatial: it takes exactly one input day (T=1)")
        super().__init__(c_strict, 1, base, depth, dropout)


class AttentionResUNet(_TemporalBase):
    """M2: day D only (T = 1); residual encoder/decoder blocks and attention-gated skips.

    Same inputs, depth and widths as M1, so M2 vs M1 tests the spatial architecture alone.
    """

    model_type = "m2_attention_res_unet"

    def __init__(self, c_strict: int, T: int = 1, base: int = 32, depth: int = 5,
                 dropout: float = 0.0):
        if T != 1:
            raise ValueError("M2 is spatial: it takes exactly one input day (T=1)")
        super().__init__(c_strict, 1, base, depth, dropout)
        self.enc = Encoder(c_strict, base, depth, block=ResBlock)
        self.dec = Decoder(base, depth, block=ResBlock, attention=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        feats = self.enc(x[:, 0])
        return self.dec(feats[-1], feats[:-1])


class ConvLSTMUNet(_TemporalBase):
    """M4: shared per-day encoder -> ConvLSTM memory -> decoder.

    Every day passes through the same encoder. A ConvLSTM runs over the T days at the two
    coarsest levels (1/32 and 1/16 resolution), where each cell sees a wide neighbourhood and the
    sequence is cheap to process; its final hidden states feed the decoder. Finer skip
    connections come from day D (the newest frame), which carries the current fire edge.
    The forget-gate bias starts at 1 so early training keeps information across days.
    """

    model_type = "m4_convlstm_unet"
    lstm_levels = 2

    def __init__(self, c_strict: int, T: int = 5, base: int = 32, depth: int = 5,
                 lstm_kernel: int = 3, dropout: float = 0.0):
        super().__init__(c_strict, T, base, depth, dropout)
        if depth < self.lstm_levels:
            raise ValueError(f"depth must be >= {self.lstm_levels}")
        self.lstm_kernel = lstm_kernel
        self.enc = Encoder(c_strict, base, depth)
        ch = self.enc.ch
        self.temporal = nn.ModuleList(
            ConvLSTM(ch[lvl], ch[lvl], kernel_size=lstm_kernel)
            for lvl in range(depth - self.lstm_levels + 1, depth + 1))
        self.dec = Decoder(base, depth, dropout=dropout)

    def spec(self) -> dict[str, object]:
        # lstm_kernel sets the ConvLSTM weight shapes; without it a checkpoint cannot be rebuilt
        return {**super().spec(), "lstm_kernel": self.lstm_kernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        b, t, c, h, w = x.shape
        feats = self.enc(x.reshape(b * t, c, h, w))  # same weights for every day
        feats = [f.reshape(b, t, *f.shape[1:]) for f in feats]
        first_lstm = self.depth - self.lstm_levels + 1
        levels = []
        for lvl, f in enumerate(feats):
            if lvl >= first_lstm:
                _, (h_last, _) = self.temporal[lvl - first_lstm](f)
                levels.append(h_last)
            else:
                levels.append(f[:, -1])
        return self.dec(levels[-1], levels[:-1])


MODEL_REGISTRY = {
    SpatialUNet.model_type: SpatialUNet,
    AttentionResUNet.model_type: AttentionResUNet,
    StackedDaysUNet.model_type: StackedDaysUNet,
    ConvLSTMUNet.model_type: ConvLSTMUNet,
}
SPATIAL_MODELS = frozenset({SpatialUNet.model_type, AttentionResUNet.model_type})
