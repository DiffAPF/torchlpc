import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from torchlpc.core import LPC
from torchlpc.recurrence import Recurrence


def get_random_biquads(cmplx=False):
    if cmplx:
        mag = torch.rand(2, dtype=torch.double)
        phase = torch.rand(2, dtype=torch.double) * 2 * torch.pi
        roots = mag * torch.exp(1j * phase)
        return torch.tensor(
            [-roots[0] - roots[1], roots[0] * roots[1]], dtype=torch.complex128
        )
    mag = torch.rand(1, dtype=torch.double)
    phase = torch.rand(1, dtype=torch.double) * torch.pi
    return torch.tensor([-mag * torch.cos(phase) * 2, mag**2], dtype=torch.double)


def create_test_inputs(batch_size, samples, cmplx=False):
    start_coeffs = get_random_biquads(cmplx)
    end_coeffs = get_random_biquads(cmplx)
    dtype = torch.complex128 if cmplx else torch.double

    A = (
        torch.stack(
            [
                torch.linspace(start_coeffs[i], end_coeffs[i], samples, dtype=dtype)
                for i in range(2)
            ]
        )
        .T.unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    x = torch.randn(batch_size, samples, dtype=dtype)
    zi = torch.randn(batch_size, 2, dtype=dtype)
    return x, A, zi


@pytest.mark.parametrize(
    "x_requires_grad",
    [True],
)
@pytest.mark.parametrize(
    "a_requires_grad",
    [True, False],
)
@pytest.mark.parametrize(
    "zi_requires_grad",
    [True, False],
)
@pytest.mark.parametrize(
    "samples",
    [32],
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
def test_low_order(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
    cmplx: bool,
    device: str,
):
    batch_size = 4
    x, A, zi = tuple(
        x.to(device) for x in create_test_inputs(batch_size, samples, cmplx)
    )
    A.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(LPC.apply, (x, A, zi), check_forward_ad=True)
    assert gradgradcheck(LPC.apply, (x, A, zi))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_float64_vs_32_cuda():
    batch_size = 4
    samples = 32
    x, A, zi = create_test_inputs(batch_size, samples)
    x = x.cuda()
    A = A.cuda()
    zi = zi.cuda()

    x32 = x.float()
    A32 = A.float()
    zi32 = zi.float()

    y64 = LPC.apply(x, A, zi)
    y32 = LPC.apply(x32, A32, zi32)

    assert torch.allclose(y64, y32.double(), atol=1e-6), torch.max(
        torch.abs(y64 - y32.double())
    )


@pytest.mark.parametrize(
    "x_requires_grad",
    [True],
)
@pytest.mark.parametrize(
    "a_requires_grad",
    [True, False],
)
@pytest.mark.parametrize(
    "zi_requires_grad",
    [True, False],
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
def test_parallel_scan(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    cmplx: bool,
    device: str,
):
    batch_size = 2
    samples = 123
    dtype = torch.complex128 if cmplx else torch.double
    x = torch.randn(batch_size, samples, dtype=dtype, device=device)
    if cmplx:
        A = torch.rand(
            batch_size, samples, dtype=torch.double, device=device
        ).sqrt() * torch.exp(
            1j
            * torch.rand(batch_size, samples, dtype=torch.double, device=device)
            * 2
            * torch.pi
        )
    else:
        A = torch.rand(batch_size, samples, dtype=dtype, device=device) * 2 - 1
    zi = torch.randn(batch_size, dtype=dtype, device=device)

    A.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(Recurrence.apply, (A, x, zi), check_forward_ad=True)
    assert gradgradcheck(Recurrence.apply, (A, x, zi))
