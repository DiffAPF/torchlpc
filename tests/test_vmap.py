import torch
import torch.nn.functional as F
from torch.func import jacfwd
import pytest
from torchlpc.core import LPC
from torchlpc.recurrence import Recurrence


from .test_grad import create_test_inputs


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
def test_vmap(device: str):
    batch_size = 4
    samples = 40
    x, A, zi = tuple(
        x.to(device) for x in create_test_inputs(batch_size, samples, False)
    )
    y = torch.randn_like(x)

    A = A[:, 0, :].clone()

    A.requires_grad = True
    zi.requires_grad = True
    x.requires_grad = True

    args = (x, A, zi)

    def func(x, A, zi):
        return F.mse_loss(LPC.apply(x, A[:, None, :].expand(-1, samples, -1), zi), y)

    jacs = jacfwd(func, argnums=tuple(range(len(args))))(*args)

    loss = func(*args)
    loss.backward()
    for jac, arg in zip(jacs, args):
        assert torch.allclose(jac, arg.grad)


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
def test_parallel_scan_vmap(device: str):
    batch_size = 3
    samples = 255
    x = torch.randn(batch_size, samples, dtype=torch.double, device=device)
    A = torch.rand(batch_size, samples, dtype=torch.double, device=device) * 2 - 1
    zi = torch.randn(batch_size, dtype=torch.double, device=device)
    y = torch.randn(batch_size, samples, dtype=torch.double, device=device)

    A.requires_grad = True
    x.requires_grad = True
    zi.requires_grad = True

    args = (x, A, zi)

    def func(x, A, zi):
        return F.mse_loss(Recurrence.apply(A, x, zi), y)

    jacs = jacfwd(func, argnums=tuple(range(len(args))))(*args)

    loss = func(*args)
    loss.backward()
    for jac, arg in zip(jacs, args):
        assert torch.allclose(jac, arg.grad)
