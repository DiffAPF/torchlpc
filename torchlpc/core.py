import warnings
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Tuple, Optional, Callable, List
from numba import jit, njit, prange, cuda, float32, float64, complex64, complex128

from . import EXTENSION_LOADED

lpc_cuda_kernel_float32: Callable = None
lpc_cuda_kernel_float64: Callable = None
lpc_cuda_kernel_complex64: Callable = None
lpc_cuda_kernel_complex128: Callable = None


for t in ["float32", "float64"]:
    exec(
        f"""@cuda.jit
def lpc_cuda_kernel_{t}(padded_y, A, B, T, order) -> None:
    sm = cuda.shared.array(shape=0, dtype={t})
    batch_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    i = tid
    b = batch_idx

    if b >= B or i >= order:
        return

    circular_idx = 0
    sm[i] = padded_y[b, i]

    for t in range(T):
        circular_idx = t % order
        a = -A[b, t, i]
        if i > circular_idx - 1:
            s = sm[circular_idx - i - 1 + order]
        else:
            s = sm[circular_idx - i - 1]
        
        v = a * s

        if i == (order - 1):
            sm[circular_idx] = v
            v = padded_y[b, t + order]
        cuda.syncthreads()
        cuda.atomic.add(sm, circular_idx, v)
        cuda.syncthreads()

        if i == (order - 1):
            padded_y[b, t + order] = sm[circular_idx]"""
    )

# separate kernel for complex type as atomic.add does not support complex types
for t, dt in zip(["complex64", "complex128"], ["float32", "float64"]):
    exec(
        f"""@cuda.jit
def lpc_cuda_kernel_{t}(padded_y, A, B, T, order) -> None:
    sm = cuda.shared.array(shape=0, dtype={dt})
    batch_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    i = tid
    b = batch_idx

    if b >= B or i >= order:
        return

    sm_real = sm[:order]
    sm_imag = sm[order:2*order]

    circular_idx = 0
    sm_real[i] = padded_y.real[b, i]
    sm_imag[i] = padded_y.imag[b, i]

    for t in range(T):
        circular_idx = t % order
        a = -A[b, t, i]
        if i > circular_idx - 1:
            s_real = sm_real[circular_idx - i - 1 + order]
            s_imag = sm_imag[circular_idx - i - 1 + order]
        else:
            s_real = sm_real[circular_idx - i - 1]
            s_imag = sm_imag[circular_idx - i - 1]
        
        v_real = a.real * s_real - a.imag * s_imag
        v_imag = a.real * s_imag + a.imag * s_real
        
        if i == (order - 1):
            sm_real[circular_idx] = v_real
            sm_imag[circular_idx] = v_imag
            v_real = padded_y.real[b, t + order]
            v_imag = padded_y.imag[b, t + order]
        cuda.syncthreads()

        cuda.atomic.add(sm_real, circular_idx, v_real)
        cuda.atomic.add(sm_imag, circular_idx, v_imag)
        cuda.syncthreads()

        if i == (order - 1):
            padded_y[b, t + order] = sm_real[circular_idx] + 1j * sm_imag[circular_idx]"""
    )


def lpc_cuda(x: torch.Tensor, A: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
    B, T, order = A.shape
    assert order <= 1024
    padded_y = torch.empty((B, T + order), dtype=x.dtype, device=x.device)
    padded_y[:, :order] = zi.flip(1)
    padded_y[:, order:] = x

    threads_per_block = order
    blocks_per_grid = B
    stream = cuda.stream()

    if x.dtype == torch.float32:
        runner = lpc_cuda_kernel_float32[
            blocks_per_grid, threads_per_block, stream, 4 * order
        ]
    elif x.dtype == torch.float64:
        runner = lpc_cuda_kernel_float64[
            blocks_per_grid, threads_per_block, stream, 8 * order
        ]
    elif x.dtype == torch.complex64:
        runner = lpc_cuda_kernel_complex64[
            blocks_per_grid, threads_per_block, stream, 8 * order
        ]
    elif x.dtype == torch.complex128:
        runner = lpc_cuda_kernel_complex128[
            blocks_per_grid, threads_per_block, stream, 16 * order
        ]
    else:
        raise NotImplementedError(f"Unsupported dtype: {x.dtype}")

    runner(cuda.as_cuda_array(padded_y), cuda.as_cuda_array(A), B, T, order)

    return padded_y[:, order:].contiguous()


@njit(parallel=True)
def lpc_np(x: np.ndarray, A: np.ndarray, zi: np.ndarray) -> np.ndarray:
    B, T = x.shape
    order = zi.shape[1]
    padded_y = np.empty((B, T + order), dtype=x.dtype)
    padded_y[:, :order] = zi[:, ::-1]
    padded_y[:, order:] = x

    for b in prange(B):
        for t in range(T):
            ref = padded_y[b, t + order]
            for i in prange(order):
                ref -= A[b, t, i] * padded_y[b, t + order - i - 1]
            padded_y[b, t + order] = ref

    return padded_y[:, order:]


class LPC(Function):
    @staticmethod
    def forward(x: torch.Tensor, A: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        if EXTENSION_LOADED:
            y = torch.ops.torchlpc.lpc(x, A, zi)
        else:
            warnings.warn(
                "Cannot find custom extension. Falling back to Numba implementation which will be deprecated in v1.0."
            )
            if x.is_cuda:
                y = lpc_cuda(x.detach(), A.detach(), zi.detach())
            else:
                y = lpc_np(
                    x.detach().cpu().numpy(),
                    A.detach().cpu().numpy(),
                    zi.detach().cpu().numpy(),
                )
                y = torch.from_numpy(y).to(x.device, x.dtype)
        return y

    @staticmethod
    def setup_context(ctx: Any, inputs: List[Any], output: Any) -> Any:
        _, A, zi = inputs
        y = output
        ctx.save_for_backward(A, zi, y)
        ctx.save_for_forward(A, zi, y)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        A, zi, y = ctx.saved_tensors
        grad_x = grad_A = grad_zi = None
        B, T, order = A.shape

        flipped_A = A.flip(2)
        padded_flipped_A = F.pad(flipped_A.transpose(1, 2), (0, order + 1))
        shifted_A = (
            padded_flipped_A.reshape(B, T + order + 1, order)[:, :-1, :]
            .reshape(B, order, T + order)
            .transpose(1, 2)
            .flip(2)
        )

        if not ctx.needs_input_grad[2]:
            shifted_A = shifted_A[:, order:, :]
            padded_grad_y = grad_y
        else:
            padded_grad_y = F.pad(grad_y.unsqueeze(1), (order, 0)).squeeze(1)

        flipped_grad_x = LPC.apply(
            padded_grad_y.flip(1),
            shifted_A.flip(1).conj_physical(),
            torch.zeros_like(zi),
        )

        if ctx.needs_input_grad[2]:
            grad_zi = flipped_grad_x[:, -order:]
            flipped_grad_x = flipped_grad_x[:, :-order]

        if ctx.needs_input_grad[0]:
            grad_x = flipped_grad_x.flip(1)

        if ctx.needs_input_grad[1]:
            valid_y = y[:, :-1]
            padded_y = torch.cat([zi.flip(1), valid_y], dim=1)

            unfolded_y = padded_y.unfold(1, order, 1).flip(2)
            grad_A = unfolded_y.conj_physical() * -flipped_grad_x.flip(1).unsqueeze(2)

        return grad_x, grad_A, grad_zi

    @staticmethod
    def jvp(
        ctx: Any, grad_x: torch.Tensor, grad_A: torch.Tensor, grad_zi: torch.Tensor
    ) -> torch.Tensor:
        A, zi, y = ctx.saved_tensors
        *_, order = A.shape

        fwd_zi = grad_zi if grad_zi is not None else torch.zeros_like(zi)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(y)

        if grad_A is not None:
            unfolded_y = (
                torch.cat([zi.flip(1), y[:, :-1]], dim=1).unfold(1, order, 1).flip(2)
            )
            fwd_A = -torch.sum(unfolded_y * grad_A, dim=2)
            fwd_x = fwd_x + fwd_A

        return LPC.apply(fwd_x, A, fwd_zi)

    @staticmethod
    def vmap(info, in_dims, *args):
        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        x, A, zi = tuple(
            map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                map(maybe_expand_bdim_at_front, args, in_dims),
            )
        )

        y = LPC.apply(x, A, zi)
        return y.reshape(info.batch_size, -1, *y.shape[1:]), 0
