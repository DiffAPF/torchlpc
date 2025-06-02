import torch
import torch.nn.functional as F
import pytest
from torchlpc.core import lpc_np, lpc_cuda


from .test_grad import create_test_inputs


@pytest.mark.parametrize(
    "samples",
    [64, 4097],
)
@pytest.mark.parametrize(
    "cmplx",
    [True, False],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_scan_equiv(samples: int, cmplx: bool, device: str):
    batch_size = 4
    x = torch.randn(
        batch_size,
        samples,
        dtype=torch.float32 if not cmplx else torch.complex64,
        device=device,
    )
    if cmplx:
        A = torch.rand(
            batch_size, samples, dtype=x.dtype, device=device
        ).sqrt() * torch.exp(
            2j
            * torch.rand(batch_size, samples, dtype=x.dtype, device=device)
            * torch.pi
        )
    else:
        A = torch.rand_like(x) * 1.8 - 0.9
    zi = torch.randn(batch_size, dtype=x.dtype, device=device)

    if device == "cuda":
        numba_y = lpc_cuda(x, -A.unsqueeze(2), zi.unsqueeze(1))
    else:
        numba_y = torch.from_numpy(
            lpc_np(
                x.cpu().numpy(),
                -A.cpu().unsqueeze(2).numpy(),
                zi.cpu().unsqueeze(1).numpy(),
            )
        )
    ext_y = torch.ops.torchlpc.scan(x, A, zi)

    assert torch.allclose(numba_y, ext_y, atol=5e-7), torch.max(
        torch.abs(numba_y - ext_y)
    ).item()


@pytest.mark.parametrize("samples", [1021, 4097])
@pytest.mark.parametrize(
    "cmplx",
    [True, False],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_lpc_equiv(samples: int, cmplx: bool, device: str):
    batch_size = 4
    x, A, zi = tuple(
        x.to(device) for x in create_test_inputs(batch_size, samples, cmplx)
    )
    if device == "cuda":
        numba_y = lpc_cuda(x, A, zi)
    else:
        numba_y = torch.from_numpy(lpc_np(x.numpy(), A.numpy(), zi.numpy()))
    ext_y = torch.ops.torchlpc.lpc(x, A, zi)

    assert torch.allclose(numba_y, ext_y)
