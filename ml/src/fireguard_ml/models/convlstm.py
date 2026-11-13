"""ConvLSTM building block for M4 (Role 3).

Input  : [B, T, C, H, W]  (frames ordered oldest -> newest)
Output : outputs [B, T, hidden, H, W] and the final (h, c), each [B, hidden, H, W]

Spatial size is preserved (odd kernel, "same" padding), which is the reason ConvLSTM is
used instead of flattening each frame into a vector.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

State = tuple[torch.Tensor, torch.Tensor]


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd so spatial size is preserved")
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size,
                               padding=kernel_size // 2)
        with torch.no_grad():
            # chunk order is (input, forget, cell, output); bias the forget gate towards remembering
            self.gates.bias[hidden_ch : 2 * hidden_ch].fill_(1.0)

    def init_state(self, x: torch.Tensor) -> State:
        b, _, h, w = x.shape
        z = x.new_zeros(b, self.hidden_ch, h, w)
        return z, z.clone()

    def forward(self, x: torch.Tensor, state: State) -> State:
        h, c = state
        i, f, g, o = torch.chunk(self.gates(torch.cat([x, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int | Sequence[int], kernel_size: int = 3,
                 num_layers: int = 1):
        super().__init__()
        hidden = [hidden_ch] * num_layers if isinstance(hidden_ch, int) else list(hidden_ch)
        if len(hidden) != num_layers:
            raise ValueError("hidden_ch list length must equal num_layers")
        self.cells = nn.ModuleList(
            ConvLSTMCell(in_ch if i == 0 else hidden[i - 1], hidden[i], kernel_size)
            for i in range(num_layers)
        )
        self.out_ch = hidden[-1]

    def forward(self, x: torch.Tensor, state: list[State] | None = None):
        if x.ndim != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(x.shape)}")
        seq = x
        finals: list[State] = []
        for li, cell in enumerate(self.cells):
            st = state[li] if state is not None else cell.init_state(seq[:, 0])
            outs = []
            for t in range(seq.shape[1]):
                st = cell(seq[:, t], st)
                outs.append(st[0])
            seq = torch.stack(outs, dim=1)
            finals.append(st)
        return seq, finals[-1]
