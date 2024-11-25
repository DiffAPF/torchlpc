import itertools
import logging
import os

import numba
import torch as tr
import torch.utils.cpp_extension
from torch.utils import benchmark
from tqdm import tqdm

from test_forward import sample_wise_lpc_scriptable
from torchlpc import sample_wise_lpc

tr.utils.cpp_extension.load(
    name="torchlpc",
    sources=["../torchlpc/csrc/torchlpc.cpp"],
    is_python_module=False,
    verbose=True,
)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

batch_sizes = [32]
n_samples = [2048]
orders = [3]
forward_funcs = [
    sample_wise_lpc,
    sample_wise_lpc_scriptable,
    tr.ops.torchlpc.forward,
    tr.ops.torchlpc.forward_batch_parallel,
]
dtype = tr.float
num_threads = 1


def main() -> None:
    tr.manual_seed(42)
    # Set intraop threads
    tr.set_num_threads(num_threads)
    # Set interop threads
    tr.set_num_interop_threads(num_threads)
    numba.set_num_threads(num_threads)
    log.info(f"numba.get_num_threads(): {numba.get_num_threads()}")

    results = []

    for bs, n, order in tqdm(itertools.product(batch_sizes, n_samples, orders)):
        x = tr.randn(bs, n, dtype=dtype)
        a = tr.randn(bs, n, order, dtype=dtype)
        zi = tr.randn(bs, order, dtype=dtype)

        x.requires_grad_(False)
        a.requires_grad_(False)
        zi.requires_grad_(False)

        for forward_func in tqdm(forward_funcs):
            globals = {
                "forward_func": forward_func,
                "x": x,
                "a": a,
                "zi": zi,
            }
            results.append(
                benchmark.Timer(
                    stmt="y = forward_func(x, a, zi)",
                    globals=globals,
                    sub_label=f"bs_{bs}__n_{n}__order_{order}__threads_{num_threads}",
                    description=forward_func.__name__,
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=0.5)
            )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
