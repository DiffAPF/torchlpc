#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <torch/script.h>
#include <torch/torch.h>

// CUDA kernel for LPC computation
template <typename scalar_t>
__global__ void lpc_cuda_kernel(scalar_t* padded_y,  // [B, T + order]
                                const scalar_t* A,   // [B, T, order]
                                int64_t B, int64_t T, int64_t order) {
    extern __shared__ char smem[];
    scalar_t* sm = reinterpret_cast<scalar_t*>(smem);

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= order) return;

    // Initialize shared memory with the first 'order' elements
    sm[i] = padded_y[b * (T + order) + i];
    __syncthreads();

    int circular_idx = 0;
    for (int t = 0; t < T; ++t) {
        circular_idx = t % order;
        scalar_t a = -A[((b * T + t) * order) + i];

        // Compute s as in the Python code
        int idx_offset = circular_idx - i - 1;
        if (i > circular_idx - 1) {
            idx_offset += order;
        }
        scalar_t s = sm[(idx_offset + order) % order];

        scalar_t v = a * s;

        if (i == order - 1) {
            sm[circular_idx] = v;
            v = padded_y[b * (T + order) + t + order];
        }
        __syncthreads();

        // Atomic add to shared memory
        atomicAdd(&sm[circular_idx], v);
        __syncthreads();

        if (i == order - 1) {
            padded_y[b * (T + order) + t + order] = sm[circular_idx];
        }
        __syncthreads();
    }
}
// CUDA kernel for complex LPC computation
template <typename scalar_t>
__global__ void lpc_cuda_kernel_complex(
    scalar_t* padded_y_real,  // [B, T + order]
    scalar_t* padded_y_imag,  // [B, T + order]
    const scalar_t* A_real,   // [B, T, order]
    const scalar_t* A_imag,   // [B, T, order]
    int64_t B, int64_t T, int64_t order) {
    extern __shared__ char smem[];
    scalar_t* sm_real = reinterpret_cast<scalar_t*>(smem);
    scalar_t* sm_imag = sm_real + order;

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= order) return;

    // Initialize shared memory with the first 'order' elements
    sm_real[i] = padded_y_real[b * (T + order) + i];
    sm_imag[i] = padded_y_imag[b * (T + order) + i];
    __syncthreads();

    int circular_idx = 0;
    for (int t = 0; t < T; ++t) {
        circular_idx = t % order;
        scalar_t a_real = -A_real[((b * T + t) * order) + i];
        scalar_t a_imag = -A_imag[((b * T + t) * order) + i];

        int idx_offset = circular_idx - i - 1;
        if (i > circular_idx - 1) {
            idx_offset += order;
        }
        int s_idx = (idx_offset + order) % order;
        scalar_t s_real = sm_real[s_idx];
        scalar_t s_imag = sm_imag[s_idx];

        // Complex multiply: v = a * s
        scalar_t v_real = a_real * s_real - a_imag * s_imag;
        scalar_t v_imag = a_real * s_imag + a_imag * s_real;

        if (i == order - 1) {
            sm_real[circular_idx] = v_real;
            sm_imag[circular_idx] = v_imag;
            v_real = padded_y_real[b * (T + order) + t + order];
            v_imag = padded_y_imag[b * (T + order) + t + order];
        }
        __syncthreads();

        atomicAdd(&sm_real[circular_idx], v_real);
        atomicAdd(&sm_imag[circular_idx], v_imag);
        __syncthreads();

        if (i == order - 1) {
            padded_y_real[b * (T + order) + t + order] = sm_real[circular_idx];
            padded_y_imag[b * (T + order) + t + order] = sm_imag[circular_idx];
        }
        __syncthreads();
    }
}

at::Tensor lpc_cuda_wrapper(const at::Tensor& x, const at::Tensor& a,
                            const at::Tensor& zi) {
    TORCH_CHECK(x.is_floating_point() || x.is_complex(),
                "Input must be floating point or complex");
    TORCH_CHECK(a.scalar_type() == x.scalar_type(),
                "Coefficients must have the same scalar type as input");
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "Initial conditions must have the same scalar type as input");

    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(zi.dim() == 2, "Initial conditions must be 2D");
    TORCH_CHECK(x.size(0) == zi.size(0),
                "Batch size of input and initial conditions must match");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    auto a_contiguous = a.contiguous();

    at::Tensor out;
    auto order = a_contiguous.size(2);
    assert(order <= 1024 && "LPC order must be less than or equal to 1024");
    auto threads_per_block = order;

    if (x.is_floating_point()) {
        out = at::cat({zi.flip(1), x}, 1).contiguous();
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lpc_cuda", [&] {
            auto padded_y = out.mutable_data_ptr<scalar_t>();
            auto A = a_contiguous.const_data_ptr<scalar_t>();
            auto B = x.size(0);
            auto T = x.size(1);

            lpc_cuda_kernel<scalar_t><<<B, threads_per_block,
                                        threads_per_block * sizeof(scalar_t)>>>(
                padded_y, A, B, T, order);
        });
    } else {
        auto out_real =
            at::cat({at::real(zi).flip(1), at::real(x)}, 1).contiguous();
        auto out_imag =
            at::cat({at::imag(zi).flip(1), at::imag(x)}, 1).contiguous();
        auto a_real = at::real(a_contiguous).contiguous();
        auto a_imag = at::imag(a_contiguous).contiguous();
        AT_DISPATCH_FLOATING_TYPES(
            out_real.scalar_type(), "lpc_cuda_complex", [&] {
                auto padded_y_real = out_real.mutable_data_ptr<scalar_t>();
                auto padded_y_imag = out_imag.mutable_data_ptr<scalar_t>();
                auto A_real = a_real.const_data_ptr<scalar_t>();
                auto A_imag = a_imag.const_data_ptr<scalar_t>();
                auto B = x.size(0);
                auto T = x.size(1);

                lpc_cuda_kernel_complex<scalar_t>
                    <<<B, threads_per_block,
                       2 * threads_per_block * sizeof(scalar_t)>>>(
                        padded_y_real, padded_y_imag, A_real, A_imag, B, T,
                        order);
            });
        out = at::view_as_complex(at::stack({out_real, out_imag}, -1));
    }
    return out.slice(1, order, out.size(1)).contiguous();
}

TORCH_LIBRARY_IMPL(torchlpc, CUDA, m) { m.impl("lpc", &lpc_cuda_wrapper); }