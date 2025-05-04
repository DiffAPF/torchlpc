#include <Python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <utility>
#include <vector>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so associated with this extension
   built from this file, so that all the TORCH_LIBRARY calls below are run.*/
PyObject *PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}
}

template <typename scalar_t>
void scan_cpu(const at::Tensor &input, const at::Tensor &weights,
              const at::Tensor &initials, const at::Tensor &output) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(initials.dim() == 1, "Initials must be 1D");
    TORCH_CHECK(weights.sizes() == input.sizes(),
                "Weights must have the same size as input");
    TORCH_CHECK(output.sizes() == input.sizes(),
                "Output must have the same size as input");
    TORCH_CHECK(initials.size(0) == input.size(0),
                "The first dimension of initials must be the same as the first "
                "dimension of input");
    TORCH_INTERNAL_ASSERT(input.device().is_cpu(), "Input must be on CPU");
    TORCH_INTERNAL_ASSERT(initials.device().is_cpu(),
                          "Initials must be on CPU");
    TORCH_INTERNAL_ASSERT(weights.device().is_cpu(), "Weights must be on CPU");
    TORCH_INTERNAL_ASSERT(output.device().is_cpu(), "Output must be on CPU");
    TORCH_INTERNAL_ASSERT(output.is_contiguous(), "Output must be contiguous");

    auto input_contiguous = input.contiguous();
    auto weights_contiguous = weights.contiguous();
    auto initials_contiguous = initials.contiguous();

    auto n_batch = input.size(0);
    auto T = input.size(1);
    auto total_size = input.numel();

    std::pair<scalar_t, scalar_t> buffer[total_size];

    const scalar_t *input_ptr = input_contiguous.const_data_ptr<scalar_t>();
    const scalar_t *initials_ptr =
        initials_contiguous.const_data_ptr<scalar_t>();
    const scalar_t *weights_ptr = weights_contiguous.const_data_ptr<scalar_t>();
    scalar_t *output_ptr = output.mutable_data_ptr<scalar_t>();

    std::transform(weights_ptr, weights_ptr + total_size, input_ptr, buffer,
                   [](const scalar_t &a, const scalar_t &b) {
                       return std::make_pair(a, b);
                   });

    at::parallel_for(0, n_batch, 1, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
            std::inclusive_scan(
                buffer + b * T, buffer + (b + 1) * T, buffer + b * T,
                [](const std::pair<scalar_t, scalar_t> &a,
                   const std::pair<scalar_t, scalar_t> &b) {
                    return std::make_pair(a.first * b.first,
                                          a.second * b.first + b.second);
                },
                std::make_pair((scalar_t)1.0, initials_ptr[b]));
        }
    });

    std::transform(
        buffer, buffer + total_size, output_ptr,
        [](const std::pair<scalar_t, scalar_t> &a) { return a.second; });
}

template <typename scalar_t>
void lpc_cpu_core(const torch::Tensor &a, const torch::Tensor &padded_out) {
    // Ensure input dimensions are correct
    TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    TORCH_CHECK(padded_out.dim() == 2, "out must be 2-dimensional");
    TORCH_CHECK(padded_out.size(0) == a.size(0),
                "Batch size of out and x must match");
    TORCH_CHECK(padded_out.size(1) == (a.size(1) + a.size(2)),
                "Time dimension of out must match x and a");
    TORCH_INTERNAL_ASSERT(a.device().is_cpu(), "a must be on CPU");
    TORCH_INTERNAL_ASSERT(padded_out.device().is_cpu(),
                          "Output must be on CPU");
    TORCH_INTERNAL_ASSERT(padded_out.is_contiguous(),
                          "Output must be contiguous");

    // Get the dimensions
    const auto B = a.size(0);
    const auto T = a.size(1);
    const auto order = a.size(2);

    auto a_contiguous = a.contiguous();

    const scalar_t *a_ptr = a_contiguous.const_data_ptr<scalar_t>();
    scalar_t *out_ptr = padded_out.mutable_data_ptr<scalar_t>();

    at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
            auto out_offset = b * (T + order) + order;
            auto a_offset = b * T * order;
            for (int64_t t = 0; t < T; t++) {
                scalar_t y = out_ptr[out_offset + t];
                for (int64_t i = 0; i < order; i++) {
                    y -= a_ptr[a_offset + t * order + i] *
                         out_ptr[out_offset + t - i - 1];
                }
                out_ptr[out_offset + t] = y;
            }
        }
    });
}

at::Tensor scan_cpu_wrapper(const at::Tensor &input, const at::Tensor &weights,
                            const at::Tensor &initials) {
    TORCH_CHECK(input.is_floating_point() || input.is_complex(),
                "Input must be floating point or complex");
    TORCH_CHECK(initials.scalar_type() == input.scalar_type(),
                "Initials must have the same scalar type as input");
    TORCH_CHECK(weights.scalar_type() == input.scalar_type(),
                "Weights must have the same scalar type as input");

    auto output = at::empty_like(input);

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        input.scalar_type(), "scan_cpu",
        [&] { scan_cpu<scalar_t>(input, weights, initials, output); });
    return output;
}

at::Tensor lpc_cpu(const at::Tensor &x, const at::Tensor &a,
                   const at::Tensor &zi) {
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

    auto out = at::cat({zi.flip(1), x}, 1).contiguous();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        x.scalar_type(), "lpc_cpu", [&] { lpc_cpu_core<scalar_t>(a, out); });
    return out.slice(1, zi.size(1), out.size(1)).contiguous();
}

TORCH_LIBRARY(torchlpc, m) {
    m.def("torchlpc::scan(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("torchlpc::lpc(Tensor a, Tensor b, Tensor c) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchlpc, CPU, m) {
    m.impl("scan", &scan_cpu_wrapper);
    m.impl("lpc", &lpc_cpu);
}
