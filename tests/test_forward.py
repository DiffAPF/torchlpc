import logging
import os
from typing import Optional, Callable

import pytest
import torch as tr
import torch.utils.cpp_extension
from torch import Tensor as T

from torchlpc import sample_wise_lpc

tr.utils.cpp_extension.load(
    name="torchlpc",
    sources=["../torchlpc/csrc/torchlpc.cpp"],
    is_python_module=False,
    verbose=True
)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# TorchScript compatible pure torch implementation of torchlpc.forward()
def sample_wise_lpc_scriptable(x: T, a: T, zi: Optional[T] = None) -> T:
    assert x.ndim == 2
    assert a.ndim == 3
    assert x.size(0) == a.size(0)
    assert x.size(1) == a.size(1)

    B, T, order = a.shape
    if zi is None:
        zi = a.new_zeros(B, order)
    else:
        assert zi.shape == (B, order)

    zi = tr.flip(zi, dims=[1])
    a_flip = tr.flip(a, dims=[2])
    padded_y = tr.cat([zi, x], dim=1)

    for t in range(T):
        prod = a_flip[:, t: t + 1] @ padded_y[:, t: t + order, None]
        prod = prod[:, 0, 0]
        padded_y[:, t + order] -= prod

    return padded_y[:, order:]


def compare_forward(forward_a: Callable[[T, T, Optional[T]], T],
                    forward_b: Callable[[T, T, Optional[T]], T],
                    bs: int,
                    n_samples: int,
                    order: int,
                    use_double: bool = True,
                    rtol: float = 1e-5,
                    atol: float = 1e-8) -> None:
    if use_double:
        dtype = tr.double
    else:
        dtype = tr.float
    x = tr.randn(bs, n_samples, dtype=dtype)
    a = tr.randn(bs, n_samples, order, dtype=dtype)
    zi = tr.randn(bs, order, dtype=dtype)
    result_a = forward_a(x, a, zi)
    result_b = forward_b(x, a, zi)
    assert tr.allclose(result_a, result_b, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "bs",
    [1, 2, 10],
)
@pytest.mark.parametrize(
    "n_samples",
    [1, 2, 2048],
)
@pytest.mark.parametrize(
    "order",
    [1, 2, 3, 6],
)
def test_forward(bs: int, n_samples: int, order: int) -> None:
    forward_a = sample_wise_lpc
    # sample_wise_lpc_scriptable
    forward_b = sample_wise_lpc_scriptable
    compare_forward(forward_a, forward_b, bs, n_samples, order)
    # CPP forward
    forward_b = tr.ops.torchlpc.forward
    compare_forward(forward_a, forward_b, bs, n_samples, order)
    # CPP forward_batch_parallel
    forward_b = tr.ops.torchlpc.forward_batch_parallel
    compare_forward(forward_a, forward_b, bs, n_samples, order)
