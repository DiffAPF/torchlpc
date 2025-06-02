# TorchLPC
[![PyPI version](https://badge.fury.io/py/torchlpc.svg)](https://badge.fury.io/py/torchlpc)

`torchlpc` provides a PyTorch implementation of the Linear Predictive Coding (LPC) filter, also known as all-pole filter.
It's fast, differentiable, and supports batched inputs with time-varying filter coefficients.

Given an input signal $`\mathbf{x} \in \mathbb{R}^T`$ and time-varying LPC coefficients $`\mathbf{A} \in \mathbb{R}^{T \times N}`$ with an order of $`N`$, the LPC filter is defined as:

$$
y_t = x_t - \sum_{i=1}^N A_{t,i} y_{t-i}.
$$

## Usage

```python

import torch
from torchlpc import sample_wise_lpc

# Create a batch of 10 signals, each with 100 time steps
x = torch.randn(10, 100)

# Create a batch of 10 sets of LPC coefficients, each with 100 time steps and an order of 3
A = torch.randn(10, 100, 3)

# Apply LPC filtering
y = sample_wise_lpc(x, A)

# Optionally, you can provide initial values for the output signal (default is 0)
zi = torch.randn(10, 3)
y = sample_wise_lpc(x, A, zi=zi)
```


## Installation

```bash
pip install torchlpc
```

or from source

```bash
pip install git+https://github.com/DiffAPF/torchlpc.git
```

## Derivation of the gradients of the LPC filter

The details of the derivation can be found in our preprints[^1][^2].
We show that, given the instataneous gradient $\frac{\partial \mathcal{L}}{\partial y_t}$ where $\mathcal{L}$ is the loss function, the gradients of the LPC filter with respect to the input signal $\bf x$ and the filter coefficients $\bf A$ can be expresssed also through a time-varying filter:

```math
\frac{\partial \mathcal{L}}{\partial x_t}
= \frac{\partial \mathcal{L}}{\partial y_t}
- \sum_{i=1}^{N} A_{t+i,i} \frac{\partial \mathcal{L}}{\partial x_{t+i}}
```

$$
\frac{\partial \mathcal{L}}{\partial \bf A}
= -\begin{vmatrix}
\frac{\partial \mathcal{L}}{\partial x_1} & 0 & \dots & 0 \\
0 & \frac{\partial \mathcal{L}}{\partial x_2} & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \frac{\partial \mathcal{L}}{\partial x_t}
\end{vmatrix}
\begin{vmatrix}
y_0 & y_{-1} & \dots & y_{-N + 1} \\
y_1 & y_0 & \dots & y_{-N + 2} \\
\vdots & \vdots & \ddots & \vdots \\
y_{T-1} & y_{T - 2} & \dots & y_{T - N}
\end{vmatrix}.
$$

### Gradients for the initial condition $`y_t|_{t \leq 0}`$

The initial conditions provide an entry point at $t=1$ for filtering, as we cannot evaluate $t=-\infty$.
Let us assume $`A_{t, :}|_{t \leq 0} = 0`$ so $`y_t|_{t \leq 0} = x_t|_{t \leq 0}`$, which also means $`\frac{\partial \mathcal{L}}{\partial y_t}|_{t \leq 0} = \frac{\partial \mathcal{L}}{\partial x_t}|_{t \leq 0}`$.
Thus, the initial condition gradients are

$$
\frac{\partial \mathcal{L}}{\partial y_t} 
= \frac{\partial \mathcal{L}}{\partial x_t}
= -\sum_{i=1-t}^{N} A_{t+i,i} \frac{\partial \mathcal{L}}{\partial x_{t+i}} \quad \text{for } -N < t \leq 0.
$$

In practice, we pad $N$ and $N \times N$ zeros to the beginning of $\frac{\partial \mathcal{L}}{\partial \bf y}$ and $\mathbf{A}$ before evaluating $\frac{\partial \mathcal{L}}{\partial \bf x}$.
The first $N$ outputs are the gradients to $`y_t|_{t \leq 0}`$ and the rest are to $`x_t|_{t > 0}`$.

### Time-invariant filtering

In the time-invariant setting, $`A_{t, i} = A_{1, i} \forall t \in [1, T]`$ and the filter is simplified to

```math
y_t = x_t - \sum_{i=1}^N a_i y_{t-i}, \mathbf{a} = A_{1,:}.
```

The gradients $`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}`$ are filtering $`\frac{\partial \mathcal{L}}{\partial \mathbf{y}}`$ with $\mathbf{a}$ backwards in time, same as in the time-varying case.
$\frac{\partial \mathcal{L}}{\partial \mathbf{a}}$ is simply doing a vector-matrix multiplication:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^T} =
-\frac{\partial \mathcal{L}}{\partial \mathbf{x}^T}
\begin{vmatrix}
y_0 & y_{-1} & \dots & y_{-N + 1} \\
y_1 & y_0 & \dots & y_{-N + 2} \\
\vdots & \vdots & \ddots & \vdots \\
y_{T-1} & y_{T - 2} & \dots & y_{T - N}
\end{vmatrix}.
$$

This algorithm is more efficient than [^3] because it only needs one pass of filtering to get the two gradients while the latter needs two.

[^1]: [Differentiable All-pole Filters for Time-varying Audio Systems](https://arxiv.org/abs/2404.07970).
[^2]: [Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis](https://arxiv.org/abs/2406.05128).
[^3]: [Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables](https://arxiv.org/abs/2306.17252).

## TODO

- [x] Use PyTorch C++ extension for faster computation.
- [x] Use native CUDA kernels for GPU computation.
- [ ] Add examples.

## Related Projects

- [torchcomp](https://github.com/DiffAPF/torchcomp): differentiable compressors that use `torchlpc` for differentiable backpropagation.
- [jaxpole](https://github.com/rodrigodzf/jaxpole): equivalent implementation in JAX by @rodrigodzf.

## Citation

If you find this repository useful in your research, please cite our work with the following BibTex entries:

```bibtex
@inproceedings{ycy2024diffapf,
    title={Differentiable All-pole Filters for Time-varying Audio Systems},
    author={Chin-Yun Yu and Christopher Mitcheltree and Alistair Carson and Stefan Bilbao and Joshua D. Reiss and György Fazekas},
    booktitle={International Conference on Digital Audio Effects (DAFx)},
    year={2024},
    pages={345--352},
}

@inproceedings{ycy2024golf,
    title     = {Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis},
    author    = {Chin-Yun Yu and György Fazekas},
    year      = {2024},
    booktitle = {Proc. Interspeech},
    pages     = {1820--1824},
    doi       = {10.21437/Interspeech.2024-1187},
}
```
