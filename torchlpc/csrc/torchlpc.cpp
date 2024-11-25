#include <torch/script.h>
#include <ATen/Parallel.h>


// Use for small T (less than a second of audio)
// TODO(cm): look into using associative scan for this
torch::Tensor torchlpc_forward(torch::Tensor x, torch::Tensor a, torch::Tensor zi) {
    // Ensure input dimensions are correct
    TORCH_CHECK(x.dim() == 2, "x must be 2-dimensional");
    TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    TORCH_CHECK(x.size(0) == a.size(0), "Batch size of x and a must match");
    TORCH_CHECK(x.size(1) == a.size(1), "Time dimension of x and a must match");

    // Get the dimensions
    const auto B = a.size(0);
    const auto T = a.size(1);
    const auto order = a.size(2);

    // Ensure the zi tensor is the correct size
    TORCH_CHECK(zi.sizes() == torch::IntArrayRef({B, order}),
                "zi must have shape (B, order)");

    // Flip zi and a to match scipy.signal.lfilter
    zi = torch::flip(zi, {1});
    a = torch::flip(a, {2});

    // Concatenate zi and x along the time dimension
    auto padded_y = torch::cat({zi, x}, 1);
    for (int64_t t = 0; t < T; ++t) {
        auto a_slice = a.slice(1, t, t + 1);
        auto y_slice = padded_y.slice(1, t, t + order).unsqueeze(2);
        auto prod = torch::matmul(a_slice, y_slice).squeeze(2);
        padded_y.slice(1, t + order, t + order + 1) -= prod;
    }

    // Remove the padding and return the result
    auto y = padded_y.slice(1, order, T + order);
    return y;
}


// Use for large T (seconds of audio) or abnormally large B
torch::Tensor torchlpc_forward_batch_parallel(torch::Tensor x,
                                              torch::Tensor a,
                                              torch::Tensor zi) {
    // Ensure input dimensions are correct
    TORCH_CHECK(x.dim() == 2, "x must be 2-dimensional");
    TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    TORCH_CHECK(x.size(0) == a.size(0), "Batch size of x and a must match");
    TORCH_CHECK(x.size(1) == a.size(1), "Time dimension of x and a must match");

    // Get the dimensions
    const auto B = a.size(0);
    const auto T = a.size(1);
    const auto order = a.size(2);

    // Ensure the zi tensor is the correct size
    TORCH_CHECK(zi.sizes() == torch::IntArrayRef({B, order}),
                "zi must have shape (B, order)");

    // Flip zi and a to match scipy.signal.lfilter
    zi = torch::flip(zi, {1});
    a = torch::flip(a, {2});

    // Concatenate zi and x along the time dimension
    auto padded_y = torch::cat({zi, x}, 1);

    // Perform the computation for each time step
    at::parallel_for(0, B, 1, [&](int64_t begin_b, int64_t end_b) {
        for (auto b = begin_b; b < end_b; ++b) {
            // The temporal loop cannot be parallelized
            for (int64_t t = 0; t < T; ++t) {
                auto a_slice = a.slice(0, b, b + 1).slice(1, t, t + 1);
                auto y_slice = padded_y.slice(0, b, b + 1)
                        .slice(1, t, t + order)
                        .unsqueeze(2);
                auto prod = torch::matmul(a_slice, y_slice).squeeze(2);
                padded_y.slice(0, b, b + 1)
                        .slice(1, t + order, t + order + 1) -= prod;
            }
        }
    });

    // Remove the padding and return the result
    auto y = padded_y.slice(1, order, T + order);
    return y;
}


TORCH_LIBRARY(torchlpc, m) {
    m.def("forward", torchlpc_forward);
    m.def("forward_batch_parallel", torchlpc_forward_batch_parallel);
}
