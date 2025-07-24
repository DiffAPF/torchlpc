#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <torch/script.h>
#include <torch/torch.h>

template <typename T>
struct recur_binary_op {
    __host__ __device__ cuda::std::tuple<T, T> operator()(
        const cuda::std::tuple<T, T> &a,
        const cuda::std::tuple<T, T> &b) const {
        auto a_first = thrust::get<0>(a);
        auto a_second = thrust::get<1>(a);
        auto b_first = thrust::get<0>(b);
        auto b_second = thrust::get<1>(b);
        return cuda::std::make_tuple(a_first * b_first,
                                     a_second * b_first + b_second);
    }
};

template <typename T>
struct output_unary_op {
    __host__ __device__ T
    operator()(const cuda::std::tuple<T, T> &state) const {
        return thrust::get<1>(state);
    }
};

template <typename scalar_t>
__host__ __device__ void compute_linear_recurrence(const scalar_t *decays,
                                                   const scalar_t *impulses,
                                                   scalar_t *out, int n_steps) {
    thrust::inclusive_scan(
        thrust::device, thrust::make_zip_iterator(decays, impulses),
        thrust::make_zip_iterator(decays + n_steps, impulses + n_steps),
        thrust::make_transform_output_iterator(out,
                                               output_unary_op<scalar_t>()),
        recur_binary_op<scalar_t>());
}

template <typename T>
struct scan_functor {
    const T *decays, *impulses;
    T *out;
    int n_steps;
    __host__ __device__ void operator()(int i) {
        compute_linear_recurrence<T>(decays + i * n_steps,
                                     impulses + i * n_steps, out + i * n_steps,
                                     n_steps);
    }
};

template <typename scalar_t>
void compute_linear_recurrence2(const scalar_t *decays,
                                const scalar_t *impulses,
                                // const scalar_t *initials,
                                scalar_t *out, int n_dims, int n_steps) {
    thrust::counting_iterator<int> it(0);
    scan_functor<scalar_t> scan_op{decays, impulses, out, n_steps};
    thrust::for_each(thrust::device, it, it + n_dims, scan_op);
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