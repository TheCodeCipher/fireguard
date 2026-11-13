import pytest
import torch

from fireguard_ml.models.convlstm import ConvLSTM, ConvLSTMCell


def test_cell_preserves_spatial_shape():
    cell = ConvLSTMCell(6, 4)
    x = torch.randn(2, 6, 10, 12)
    h, c = cell(x, cell.init_state(x))
    assert h.shape == c.shape == (2, 4, 10, 12)


def test_sequence_output_shapes():
    m = ConvLSTM(6, 4)
    out, (h, c) = m(torch.randn(2, 5, 6, 10, 12))
    assert out.shape == (2, 5, 4, 10, 12)
    assert h.shape == c.shape == (2, 4, 10, 12)
    assert torch.allclose(out[:, -1], h)


def test_single_timestep_and_two_layers():
    out, (h, _) = ConvLSTM(3, [4, 5], num_layers=2)(torch.randn(1, 1, 3, 8, 8))
    assert out.shape == (1, 1, 5, 8, 8) and h.shape == (1, 5, 8, 8)


def test_representative_feature_map_shape():
    # bottleneck of a 320x352 padded frame at depth 5 is 10x11
    out, _ = ConvLSTM(16, 16)(torch.randn(1, 5, 16, 10, 11))
    assert out.shape == (1, 5, 16, 10, 11)


def test_gradients_reach_all_timesteps():
    x = torch.randn(1, 4, 3, 6, 6, requires_grad=True)
    out, _ = ConvLSTM(3, 4)(x)
    out[:, -1].sum().backward()
    assert all(x.grad[:, t].abs().sum() > 0 for t in range(4))


def test_rejects_even_kernel_and_bad_rank():
    with pytest.raises(ValueError):
        ConvLSTMCell(3, 4, kernel_size=2)
    with pytest.raises(ValueError):
        ConvLSTM(3, 4)(torch.randn(1, 3, 8, 8))
