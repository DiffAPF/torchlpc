#include <torch/script.h>
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <utility>

template <typename scalar_t>
at::Tensor scan_cpu(const at::Tensor &input, const at::Tensor &initials, const at::Tensor &weights)
{
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(initials.dim() == 1, "Initials must be 1D");
    TORCH_CHECK(weights.sizes() == input.sizes(), "Weights must have the same size as input");
    TORCH_CHECK(initials.size(0) == input.size(0), "The first dimension of initials must be the same as the first dimension of input");
    // TORCH_INTERNAL_ASSERT(input.scalar_type() == at::kFloat, "Input must be float");
    // TORCH_INTERNAL_ASSERT(initials.scalar_type() == at::kFloat, "Initials must be float");
    // TORCH_INTERNAL_ASSERT(weights.scalar_type() == at::kFloat, "Weights must be float");
    TORCH_INTERNAL_ASSERT(input.device().is_cpu(), "Input must be on CPU");
    TORCH_INTERNAL_ASSERT(initials.device().is_cpu(), "Initials must be on CPU");
    TORCH_INTERNAL_ASSERT(weights.device().is_cpu(), "Weights must be on CPU");
    TORCH_INTERNAL_ASSERT(input.is_contiguous(), "Input must be contiguous");
    TORCH_INTERNAL_ASSERT(initials.is_contiguous(), "Initials must be contiguous");
    TORCH_INTERNAL_ASSERT(weights.is_contiguous(), "Weights must be contiguous");

    auto n_batch = input.size(0);
    auto T = input.size(1);
    auto total_size = input.numel();

    std::array<std::pair<scalar_t, scalar_t>, total_size> buffer;
    at::Tensor output = at::empty_like(input);

    const scalar_t *input_ptr = input.data_ptr<scalar_t>();
    const scalar_t *initials_ptr = initials.data_ptr<scalar_t>();
    const scalar_t *weights_ptr = weights.data_ptr<scalar_t>();
    scalar_t *output_ptr = output.data_ptr<scalar_t>();

    std::transform(weights_ptr, weights_ptr + total_size, input_ptr, buffer.begin(), std::make_pair<scalar_t, scalar_t>);

    at::parallel_for(0, n_batch, 1, [buffer, T, initials_ptr](int64_t start, int64_t end)
                     {
        for (auto b = start; b < end; b++)
        {
            std::inclusive_scan(
            buffer.begin() + b * T, 
            buffer.begin() + (b + 1) * T, 
            buffer.begin() + b * T, 
            [](std::pair<scalar_t, scalar_t> &a, const std::pair<scalar_t, scalar_t> &b) {
                return std::make_pair(a.first * b.first, a.second * b.first + b.second);
            }, std::make_pair(1.0, initials_ptr[b]));
        } });

    std::transform(buffer.begin(), buffer.end(), output_ptr, [](const std::pair<scalar_t, scalar_t> &a)
                   { return a.second; });

    return output;
}

at::Tensor scan_cpu_wrapper(const at::Tensor &input, const at::Tensor &initials, const at::Tensor &weights)
{
    TORCH_CHECK(input.is_floating_point() || input.is_complex(), "Input must be floating point or complex");
    TORCH_CHECK(initials.is_floating_point() || initials.is_complex(), "Initials must be floating point or complex");
    TORCH_CHECK(weights.is_floating_point() || weights.is_complex(), "Weights must be floating point or complex");

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "scan_cpu", [&]
                                           { return scan_cpu<scalar_t>(input, initials, weights); });
}

TORCH_LIBRARY(torchlpc, m)
{
    m.def("torchlpc::scan_cpu(Tensor input, Tensor initials, Tensor weights) -> Tensor", &scan_cpu<float>);
}

TORCH_LIBRARY_IMPL(torchlpc, CPU, m)
{
    m.impl("scan_cpu", &scan_cpu_wrapper);
}