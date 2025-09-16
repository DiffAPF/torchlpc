import torch
from typing import Optional, Union, Tuple
from pathlib import Path
import warnings

# so_files = list(Path(__file__).parent.glob("_C*.so"))
# # assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
# if len(so_files) == 1:
#     torch.ops.load_library(so_files[0])
#     EXTENSION_LOADED = True
# elif len(so_files) > 1:
#     raise ValueError(f"Expected one _C*.so file, found {len(so_files)}")
# else:
#     warnings.warn("No _C*.so file found. Custom extension not loaded.")
#     EXTENSION_LOADED = False

try:
    from . import _C

    EXTENSION_LOADED = True
except ImportError:
    EXTENSION_LOADED = False
    warnings.warn("Custom extension not loaded. Falling back to Numba implementation.")

from .core import LPC
from .recurrence import Recurrence

__all__ = ["sample_wise_lpc", "linear_recurrence"]


def linear_recurrence(
    a: torch.Tensor, x: torch.Tensor, zi: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute linear recurrence using the recurrence relation:
    y[n] = x[n] + a[n] * (x[n-1] + a[n-1] * (x[n-2] + a[n-2] * (...(x[0] + a[0] * zi)...)))

    Args:
        a (torch.Tensor): Coefficients of the recurrence relation.
        x (torch.Tensor): Input signal.
        zi (torch.Tensor, optional): Initial conditions. Defaults to zero if not provided.

    Shape:
        - a: :math:`(B, T)`
        - x: :math:`(B, T)`
        - zi: :math:`(B,)`

    Returns:
        Output signal with the same shape as x.
    """
    assert a.shape == x.shape
    assert a.ndim == 2
    assert x.ndim == 2
    B, _ = a.shape

    if zi is None:
        zi = a.new_zeros(B)
    else:
        assert zi.shape == (B,)

    return Recurrence.apply(a, x, zi)  # type: ignore


def sample_wise_lpc(
    x: torch.Tensor,
    a: torch.Tensor,
    zi: Optional[torch.Tensor] = None,
    return_zf: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute LPC filtering sample-wise.

    Args:
        x (torch.Tensor): Input signal.
        a (torch.Tensor): LPC coefficients.
        zi (torch.Tensor): Initial conditions.
        return_zf (bool): If True, return the final filter delay values. Defaults to False.

    Shape:
        - x: :math:`(B, T)`
        - a: :math:`(B, T, order)`
        - zi: :math:`(B, order)`

    Returns:
        Filtered signal with the same shape as x if `return_zf` is False.
        If `return_zf` is True, returns a tuple of the filtered signal and the final delay values.
    """
    assert x.shape[0] == a.shape[0]
    assert x.shape[1] == a.shape[1]
    assert x.ndim == 2
    assert a.ndim == 3

    B, T, order = a.shape
    if zi is None:
        zi = a.new_zeros(B, order)
    else:
        assert zi.shape == (B, order)

    if order == 1:
        y = linear_recurrence(-a.squeeze(2), x, zi.squeeze(1))
    else:
        y = LPC.apply(x, a, zi)

    if return_zf:
        return y, y[:, -order:].flip(1)
    return y
