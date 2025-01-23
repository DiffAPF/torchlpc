#include <torch/script.h>
#include <torch/torch.h>
#include <algorithm>
#include <vector>
#include <utility>

template <typename scalar_t>
void scan_cpu(const at::Tensor &input, const at::Tensor &weights, const at::Tensor &initials, at::Tensor &output)
{
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(initials.dim() == 1, "Initials must be 1D");
    TORCH_CHECK(weights.sizes() == input.sizes(), "Weights must have the same size as input");
    TORCH_CHECK(output.sizes() == input.sizes(), "Output must have the same size as input");
    TORCH_CHECK(initials.size(0) == input.size(0), "The first dimension of initials must be the same as the first dimension of input");
    TORCH_INTERNAL_ASSERT(input.device().is_cpu(), "Input must be on CPU");
    TORCH_INTERNAL_ASSERT(initials.device().is_cpu(), "Initials must be on CPU");
    TORCH_INTERNAL_ASSERT(weights.device().is_cpu(), "Weights must be on CPU");
    TORCH_INTERNAL_ASSERT(output.device().is_cpu(), "Output must be on CPU");
    TORCH_INTERNAL_ASSERT(input.is_contiguous(), "Input must be contiguous");
    TORCH_INTERNAL_ASSERT(weights.is_contiguous(), "Weights must be contiguous");
    TORCH_INTERNAL_ASSERT(output.is_contiguous(), "Output must be contiguous");

    auto n_batch = input.size(0);
    auto T = input.size(1);
    auto total_size = input.numel();

    std::pair<scalar_t, scalar_t> buffer[total_size];

    const scalar_t *input_ptr = input.data_ptr<scalar_t>();
    const scalar_t *initials_ptr = initials.data_ptr<scalar_t>();
    const scalar_t *weights_ptr = weights.data_ptr<scalar_t>();
    scalar_t *output_ptr = output.data_ptr<scalar_t>();

    std::transform(weights_ptr, weights_ptr + total_size, input_ptr, buffer,
                   [](const scalar_t &a, const scalar_t &b)
                   { return std::make_pair(a, b); });

    at::parallel_for(0, n_batch, 1, [&](int64_t start, int64_t end)
                     {
        for (auto b = start; b < end; b++)
        {
            std::inclusive_scan(
                buffer + b * T,
                buffer + (b + 1) * T,
                buffer + b * T,
                [](const std::pair<scalar_t, scalar_t> &a, const std::pair<scalar_t, scalar_t> &b) {
                    return std::make_pair(a.first * b.first, a.second * b.first + b.second);
                },
                std::make_pair((scalar_t)1.0, initials_ptr[b]));
        } });

    std::transform(buffer, buffer + total_size, output_ptr, [](const std::pair<scalar_t, scalar_t> &a)
                   { return a.second; });
}

void scan_cpu_wrapper(const at::Tensor &input, const at::Tensor &weights, const at::Tensor &initials, at::Tensor &output)
{
    TORCH_CHECK(input.is_floating_point() || input.is_complex(), "Input must be floating point or complex");
    TORCH_CHECK(initials.scalar_type() == input.scalar_type(), "Initials must have the same scalar type as input");
    TORCH_CHECK(weights.scalar_type() == input.scalar_type(), "Weights must have the same scalar type as input");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "Output must have the same scalar type as input");

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "scan_cpu", [&]
                                           { scan_cpu<scalar_t>(input, weights, initials, output); });
}

TORCH_LIBRARY(torchlpc, m)
{
    m.def("torchlpc::scan_cpu(Tensor a, Tensor b, Tensor c, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(torchlpc, CPU, m)
{
    m.impl("scan_cpu", &scan_cpu_wrapper);
}
