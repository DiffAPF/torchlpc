#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <torch/script.h>
#include <torch/torch.h>

template <typename T>
struct recur_binary_op {
    __host__ __device__ cuda::std::pair<T, T> operator()(
        const cuda::std::pair<T, T> &a, const cuda::std::pair<T, T> &b) const {
        return cuda::std::make_pair(a.first * b.first,
                                    a.second * b.first + b.second);
    }
};

template <typename T>
struct scan_functor {
    thrust::device_ptr<cuda::std::pair<T, T>> data;
    int n_steps;
    __host__ __device__ void operator()(int i) const {
        thrust::inclusive_scan(thrust::device, data + i * n_steps,
                               data + (i + 1) * n_steps, data + i * n_steps,
                               recur_binary_op<T>());
    }
};

template <typename scalar_t>
void compute_linear_recurrence(const scalar_t *decays, const scalar_t *impulses,
                               scalar_t *out, int n_steps) {
    thrust::device_vector<cuda::std::pair<scalar_t, scalar_t>> pairs(n_steps);

    // Initialize input_states and output_states
    thrust::transform(
        thrust::device, decays, decays + n_steps, impulses, pairs.begin(),
        [] __host__ __device__(const scalar_t &decay, const scalar_t &impulse) {
            return cuda::std::make_pair(decay, impulse);
        });

    // auto initial_state_pair = cuda::std::make_pair(0.0, initial_state[0]);

    recur_binary_op<scalar_t> binary_op;

    thrust::inclusive_scan(thrust::device, pairs.begin(), pairs.end(),
                           pairs.begin(), binary_op);

    thrust::transform(thrust::device, pairs.begin(), pairs.end(), out,
                      [] __host__ __device__(
                          const cuda::std::pair<scalar_t, scalar_t> &state) {
                          return state.second;
                      });
}

template <typename scalar_t>
void compute_linear_recurrence2(const scalar_t *decays,
                                const scalar_t *impulses,
                                // const scalar_t *initials,
                                scalar_t *out, int n_dims, int n_steps) {
    thrust::device_vector<cuda::std::pair<scalar_t, scalar_t>> pairs(n_steps *
                                                                     n_dims);
    thrust::transform(
        thrust::device, decays, decays + n_steps * n_dims, impulses,
        pairs.begin(),
        [] __host__ __device__(const scalar_t &decay, const scalar_t &impulse) {
            return cuda::std::make_pair(decay, impulse);
        });

    recur_binary_op<scalar_t> binary_op;
    thrust::counting_iterator<int> it(0);
    scan_functor<scalar_t> scan_op{pairs.data(), n_steps};

    thrust::for_each(thrust::device, it, it + n_dims, scan_op);

    thrust::transform(thrust::device, pairs.begin(), pairs.end(), out,
                      [] __host__ __device__(
                          const cuda::std::pair<scalar_t, scalar_t> &state) {
                          return state.second;
                      });
}

at::Tensor scan_cuda_wrapper(const at::Tensor &input, const at::Tensor &weights,
                             const at::Tensor &initials) {
    TORCH_CHECK(input.is_floating_point() || input.is_complex(),
                "Input must be floating point or complex");
    TORCH_CHECK(initials.scalar_type() == input.scalar_type(),
                "Initials must have the same scalar type as input");
    TORCH_CHECK(weights.scalar_type() == input.scalar_type(),
                "Weights must have the same scalar type as input");

    auto input_contiguous =
        at::cat({initials.unsqueeze(1), input}, 1).contiguous();
    auto weights_contiguous =
        at::cat({at::zeros_like(initials.unsqueeze(1)), weights}, 1)
            .contiguous();
    auto output = at::empty_like(input_contiguous);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        input.scalar_type(), "compute_linear_recurrence", [&] {
            compute_linear_recurrence<scalar_t>(
                weights_contiguous.const_data_ptr<scalar_t>(),
                input_contiguous.const_data_ptr<scalar_t>(),
                output.mutable_data_ptr<scalar_t>(), input_contiguous.numel());
            // compute_linear_recurrence2<scalar_t>(
            //     weights_contiguous.const_data_ptr<scalar_t>(),
            //     input_contiguous.const_data_ptr<scalar_t>(),
            //     // initials.const_data_ptr<scalar_t>(),
            //     output.mutable_data_ptr<scalar_t>(),
            //     input_contiguous.size(0), input_contiguous.size(1));
        });
    return output.slice(1, 1, output.size(1))
        .contiguous();  // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(torchlpc, CUDA, m) { m.impl("scan", &scan_cuda_wrapper); }