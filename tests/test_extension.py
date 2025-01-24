import torch
import torch.nn.functional as F
import pytest
from torchlpc.core import lpc_np


from .test_grad import create_test_inputs


@pytest.mark.parametrize(
    "samples",
    [64, 4097],
)
@pytest.mark.parametrize(
    "cmplx",
    [True, False],
)
def test_scan_cpu_equiv(samples: int, cmplx: bool):
    batch_size = 4
    x = torch.randn(
        batch_size, samples, dtype=torch.float32 if not cmplx else torch.complex64
    )
    A = torch.rand_like(x) * 1.8 - 0.9
    zi = torch.randn(batch_size, dtype=x.dtype)

    numba_y = torch.from_numpy(
        lpc_np(
            x.cpu().numpy(),
            -A.cpu().unsqueeze(2).numpy(),
            zi.cpu().unsqueeze(1).numpy(),
        )
    )
    ext_y = torch.ops.torchlpc.scan_cpu(x, A, zi)

    assert torch.allclose(numba_y, ext_y)


@pytest.mark.parametrize(
    "samples",
    [1024],
)
@pytest.mark.parametrize(
    "cmplx",
    [True, False],
)
def test_lpc_cpu_equiv(samples: int, cmplx: bool):
    batch_size = 4
    x, A, zi = tuple(
        x.to("cpu") for x in create_test_inputs(batch_size, samples, cmplx)
    )
    numba_y = torch.from_numpy(lpc_np(x.numpy(), A.numpy(), zi.numpy()))
    ext_y = torch.ops.torchlpc.lpc_cpu(x, A, zi)

    assert torch.allclose(numba_y, ext_y)
