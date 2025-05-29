import torch
import torch.nn.functional as F
from torch.autograd import Function
from numba import cuda
from typing import Tuple, Optional, Any, List

from .parallel_scan import compute_linear_recurrence, WARPSIZE
from .core import lpc_cuda, lpc_np
from . import EXTENSION_LOADED

if EXTENSION_LOADED:
    lpc_cuda_runner = torch.ops.torchlpc.lpc
    lpc_cpu_runner = torch.ops.torchlpc.lpc

    scan_cuda_runner = torch.ops.torchlpc.scan
    scan_cpu_runner = torch.ops.torchlpc.scan
else:
    lpc_cuda_runner = lpc_cuda
    lpc_cpu_runner = lambda x, A, zi: torch.from_numpy(
        lpc_np(x.detach().numpy(), A.detach().numpy(), zi.detach().numpy())
    )

    scan_cuda_runner = lambda impulse, decay, initial_state: (
        lambda out: (
            out,
            compute_linear_recurrence(
                cuda.as_cuda_array(decay.detach()),
                cuda.as_cuda_array(impulse.detach()),
                cuda.as_cuda_array(initial_state.detach()),
                cuda.as_cuda_array(out),
                decay.shape[0],
                decay.shape[1],
            ),
        )
    )(torch.empty_like(impulse))[0]
    scan_cpu_runner = lambda impulse, decay, initial_state: torch.from_numpy(
        lpc_np(
            impulse.detach().numpy(),
            -decay.unsqueeze(2).detach().numpy(),
            initial_state.unsqueeze(1).detach().numpy(),
        )
    )


def _cuda_recurrence(
    impulse: torch.Tensor, decay: torch.Tensor, initial_state: torch.Tensor
) -> torch.Tensor:
    n_dims, n_steps = decay.shape
    if n_dims * WARPSIZE < n_steps:
        runner = scan_cuda_runner
    else:
        runner = lambda impulse, decay, initial_state: lpc_cuda_runner(
            impulse, -decay.unsqueeze(2), initial_state.unsqueeze(1)
        )
    return runner(impulse, decay, initial_state)


def _cpu_recurrence(
    impulse: torch.Tensor, decay: torch.Tensor, initial_state: torch.Tensor
) -> torch.Tensor:
    num_threads = torch.get_num_threads()
    n_dims, _ = decay.shape
    # This is just a rough estimation of the computational cost
    if EXTENSION_LOADED and min(n_dims, num_threads) < num_threads / 3:
        runner = scan_cpu_runner
    else:
        runner = lambda impulse, decay, initial_state: lpc_cpu_runner(
            impulse, -decay.unsqueeze(2), initial_state.unsqueeze(1)
        )
    return runner(impulse, decay, initial_state)


class Recurrence(Function):
    @staticmethod
    def forward(
        decay: torch.Tensor,
        impulse: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        if decay.is_cuda:
            out = _cuda_recurrence(impulse, decay, initial_state)
        else:
            out = _cpu_recurrence(impulse, decay, initial_state)
        return out

    @staticmethod
    def setup_context(ctx: Any, inputs: List[Any], output: Any) -> Any:
        decay, _, initial_state = inputs
        ctx.save_for_backward(decay, initial_state, output)
        ctx.save_for_forward(decay, initial_state, output)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        decay, initial_state, out = ctx.saved_tensors
        grad_decay = grad_impulse = grad_initial_state = None
        n_dims, _ = decay.shape

        padded_decay = F.pad(decay.unsqueeze(1), (0, 1)).squeeze(1)
        if ctx.needs_input_grad[2]:
            padded_grad_out = F.pad(grad_out.unsqueeze(1), (1, 0)).squeeze(1)
        else:
            padded_grad_out = grad_out
            padded_decay = padded_decay[:, 1:]

        init = padded_grad_out.new_zeros(n_dims)
        flipped_grad_impulse = Recurrence.apply(
            padded_decay.flip(1).conj_physical(),
            padded_grad_out.flip(1),
            init,
        )

        if ctx.needs_input_grad[2]:
            grad_initial_state = flipped_grad_impulse[:, -1]
            flipped_grad_impulse = flipped_grad_impulse[:, :-1]

        if ctx.needs_input_grad[1]:
            grad_impulse = flipped_grad_impulse.flip(1)

        if ctx.needs_input_grad[0]:
            valid_out = out[:, :-1]
            padded_out = torch.cat([initial_state.unsqueeze(1), valid_out], dim=1)
            grad_decay = padded_out.conj_physical() * flipped_grad_impulse.flip(1)

        return grad_decay, grad_impulse, grad_initial_state

    @staticmethod
    def jvp(
        ctx: Any,
        grad_decay: torch.Tensor,
        grad_impulse: torch.Tensor,
        grad_initial_state: torch.Tensor,
    ) -> torch.Tensor:
        decay, initial_state, out = ctx.saved_tensors

        fwd_initial_state = (
            grad_initial_state
            if grad_initial_state is not None
            else torch.zeros_like(initial_state)
        )
        fwd_impulse = (
            grad_impulse if grad_impulse is not None else torch.zeros_like(out)
        )

        if grad_decay is not None:
            concat_out = torch.cat([initial_state.unsqueeze(1), out[:, :-1]], dim=1)
            fwd_decay = concat_out * grad_decay
            fwd_impulse = fwd_impulse + fwd_decay

        return Recurrence.apply(decay, fwd_impulse, fwd_initial_state)

    @staticmethod
    def vmap(info, in_dims, *args):
        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        decay, impulse, initial_state = tuple(
            map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                map(maybe_expand_bdim_at_front, args, in_dims),
            )
        )

        out = Recurrence.apply(decay, impulse, initial_state)
        return out.reshape(info.batch_size, -1, *out.shape[1:]), 0


RecurrenceCUDA = Recurrence
