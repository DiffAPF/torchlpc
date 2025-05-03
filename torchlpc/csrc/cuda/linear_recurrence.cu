#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <torch/script.h>
#include <torch/torch.h>

#define CEIL_DIV(x, y) ((x + y - 1) / y)

#define gpuErrChk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
    }
}

__device__ int2 divide_work(int n_jobs, int n_workers, int worker_idx) {
    // Each worker will do a continuous slice of either n_jobs / n_workers
    // or ceil_div(n_jobs, n_workers). The return value is an int2 representing
    // a half open interval of jobs for the worker to perform (perform jobs
    // i for a <= i < b)

    int cd = CEIL_DIV(n_jobs, n_workers);
    int d = n_jobs / n_workers;

    int doing_cd = n_jobs % n_workers;

    int2 retval;
    if (worker_idx < doing_cd) {
        retval.x = worker_idx * cd;
        retval.y = retval.x + cd;
    } else {
        retval.x = doing_cd * cd + (worker_idx - doing_cd) * d;
        retval.y = retval.x + d;
    }

    return retval;
}

__device__ int2 compute_warp_start_stop(int block_idx, int warp_idx,
                                        int n_blocks, int n_steps) {
    int2 block_ss = divide_work(n_steps, n_blocks, block_idx);
    int block_start = block_ss.x;
    int block_stop = block_ss.y;
    int block_jobs = block_stop - block_start;

    int2 warp_ss = divide_work(block_jobs, 32, warp_idx);
    int warp_start = block_start + warp_ss.x;
    int warp_stop = block_start + warp_ss.y;

    int2 retval;
    retval.x = warp_start;
    retval.y = warp_stop;
    return retval;
}

// decay storage, h_storage:
//   each a n_dims x 33 x n_blocks matrix on GPU with 33rd column for block
//   reduction
template <typename scalar_t>
__global__ void reduction_kernel(const scalar_t *decays,
                                 const scalar_t *impulses,
                                 const scalar_t *initial_state,
                                 scalar_t *_decay_storage, scalar_t *_h_storage,
                                 int n_dims, int n_steps) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    scalar_t *decay_storage = &_decay_storage[blockIdx.x * 33 * n_dims];
    scalar_t *h_storage = &_h_storage[blockIdx.x * 33 * n_dims];

    int2 start_stop =
        compute_warp_start_stop(blockIdx.x, warp, gridDim.x, n_steps);
    int warp_start = start_stop.x;
    int warp_stop = start_stop.y;

    /*
     * Reduce within warps.
     * After this loop exits, the storage arrays should contain the reduction
     * from warp_start to warp_stop (including initial state) at index
     * (feature_idx, warp, block).
     */
    for (int i = lane; i < n_dims; i += 32) {
        scalar_t cum_decay = static_cast<scalar_t>(1.0);
        scalar_t h = static_cast<scalar_t>(0.0);
        if (blockIdx.x == 0 && warp == 0 && initial_state != NULL) {
            h = initial_state[i];
        }

        for (int t = warp_start; t < warp_stop; t++) {
            cum_decay *= decays[i + t * n_dims];
            h = decays[i + t * n_dims] * h + impulses[i + t * n_dims];
        }

        // TODO: store into shared memory, work in shared memory sized blocks
        // store into global memory
        decay_storage[i + warp * n_dims] = cum_decay;
        h_storage[i + warp * n_dims] = h;
    }

    __syncthreads();

    /*
     * Reduce over warps.
     * After this loop exits, the storage arrays should contain the reduction
     * from block_start to block_finish (including initial state) at index
     * (feature_idx, 32, block).
     */
    // TODO: parallel reduction (or scan). Need to worry about changing the warp
    //       reduction values (as I use them again later)
    for (int i = lane + 32 * warp; i < n_dims; i += blockDim.x) {
        scalar_t cum_decay = static_cast<scalar_t>(1.0);
        scalar_t h = static_cast<scalar_t>(0.0);
        for (int t = 0; t < 32; t++) {
            cum_decay *= decay_storage[i + t * n_dims];
            h = decay_storage[i + t * n_dims] * h + h_storage[i + t * n_dims];
        }
        decay_storage[i + 32 * n_dims] = cum_decay;
        h_storage[i + 32 * n_dims] = h;
    }
}

template <typename scalar_t>
__global__ void block_scan_kernel(scalar_t *decay_storage, scalar_t *h_storage,
                                  int n_dims, int n_blocks) {
    /*
     * Scan over blocks.
     * After this loop exits, the storage arrays should contain the cumulative
     * sum from block_idx 0 to i (inclusive) at index (feature_idx, 32, i) This
     * means (feature_idx, 32, 2) contains the reduction of blocks 0, 1, and 2.
     */
    // TODO: parallel scan (tricky because number of blocks isn't necessarily
    //       smaller than number of warps that can fit in a single block)
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_dims;
         i += blockDim.x * gridDim.x) {
        for (int t = 1; t < n_blocks; t++) {
            int cur_idx = i + 32 * n_dims + t * 33 * n_dims;
            int prev_idx = i + 32 * n_dims + (t - 1) * 33 * n_dims;

            // TODO: remove unneccessary reads from global memory (prev_idx
            // accesses)
            h_storage[cur_idx] = decay_storage[cur_idx] * h_storage[prev_idx] +
                                 h_storage[cur_idx];
            decay_storage[cur_idx] *= decay_storage[prev_idx];
        }
    }
}

template <typename scalar_t>
__global__ void warp_scan_kernel(const scalar_t *decays,
                                 const scalar_t *impulses,
                                 const scalar_t *initial_state, scalar_t *out,
                                 scalar_t *decay_storage, scalar_t *h_storage,
                                 int n_dims, int n_steps) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    // Note: Due to the index ordering of the storage arrays, the following
    // indices are equivalent:
    //
    // i + (t - 1) * n_dims + blockIdx.x * 33 * n_dims
    // i + 32 * n_dims + (blockIdx.x - 1) * 33 * n_dims
    //
    // when t is 0. This means something that looks like negative indexing
    // (t-1) can be used to safely access the stored value for the previous
    // warp (even if the previous warp belonged to the previous block).

    /*
     * Scan over warps.
     * After this loop executes, the storage arrays should contain the
     * cumulative sum from the beginning of sequence (including initial
     * condition) up to and including the indexed warp and block.
     */
    // TODO: parallel scan
    for (int i = lane + 32 * warp; i < n_dims; i += blockDim.x) {
        for (int t = 0; t < 32; t++) {
            if (t == 0 && blockIdx.x == 0) {
                // the reduction over warp 0 (including initial condition) is
                // correct val for scan, so there's no work to do
                continue;
            }

            int cur_idx = i + t * n_dims + blockIdx.x * 33 * n_dims;
            int prev_idx = i + (t - 1) * n_dims + blockIdx.x * 33 * n_dims;
            h_storage[cur_idx] = decay_storage[cur_idx] * h_storage[prev_idx] +
                                 h_storage[cur_idx];
            decay_storage[cur_idx] *= decay_storage[prev_idx];
        }
    }

    __syncthreads();

    int2 start_stop =
        compute_warp_start_stop(blockIdx.x, warp, gridDim.x, n_steps);
    int warp_start = start_stop.x;
    int warp_stop = start_stop.y;

    /*
     * Scan within warps.
     * This loop writes to the output array. Each warp reads in it's initial
     * state (either from the "initial_state" or the storage arrays) and then
     * writes to output for indices warp_start up to warp_stop.
     */
    for (int i = lane; i < n_dims; i += 32) {
        scalar_t h = static_cast<scalar_t>(0.0);
        if (blockIdx.x == 0 && warp == 0) {
            if (initial_state != NULL) {
                h = initial_state[i];
            }
        } else {
            h = h_storage[i + (warp - 1) * n_dims + blockIdx.x * 33 * n_dims];
        }

        for (int t = warp_start; t < warp_stop; t++) {
            h = decays[i + t * n_dims] * h + impulses[i + t * n_dims];
            out[i + t * n_dims] = h;
        }
    }
}

/*
 * This is the main method for the prefix sum kernels.
 * decays, impulses, out:
 *   each a n_dims x n_steps column major matrix located on GPU
 * initial_state:
 *   array of size n_dims located on GPU
 */
template <typename scalar_t>
void compute_linear_recurrence(const scalar_t *decays, const scalar_t *impulses,
                               const scalar_t *initial_state, scalar_t *out,
                               int n_dims, int n_steps) {
    // TODO: query
    int n_SMs = 15;
    int n_blocks_per_sm = 2;

    // we want at least 32 elements per block, but no reason to run
    // with more than the maximum number of concurrent blocks
    int n_blocks = min(CEIL_DIV(n_steps, 32), n_SMs * n_blocks_per_sm);

    // TODO: make user pass in working memory? This allows integration
    //       with CNMeM (used by Theano)
    int reduction_mem_sz = 2 * n_blocks * 33 * n_dims * sizeof(float);
    scalar_t *d_reduction_mem;
    gpuErrChk(cudaMalloc(&d_reduction_mem, reduction_mem_sz));
    scalar_t *d_decay_storage = &d_reduction_mem[0 * n_blocks * 33 * n_dims];
    scalar_t *d_h_storage = &d_reduction_mem[1 * n_blocks * 33 * n_dims];

    // TODO: run kernels on non-default stream?
    reduction_kernel<<<n_blocks, 1024>>>(decays, impulses, initial_state,
                                         d_decay_storage, d_h_storage, n_dims,
                                         n_steps);

    block_scan_kernel<<<n_blocks, 1024>>>(d_decay_storage, d_h_storage, n_dims,
                                          n_blocks);

    warp_scan_kernel<<<n_blocks, 1024>>>(decays, impulses, initial_state, out,
                                         d_decay_storage, d_h_storage, n_dims,
                                         n_steps);

    gpuErrChk(cudaFree(d_reduction_mem));
}

at::Tensor scan_cuda_wrapper(const at::Tensor &input, const at::Tensor &weights,
                             const at::Tensor &initials) {
    TORCH_CHECK(input.is_floating_point() || input.is_complex(),
                "Input must be floating point or complex");
    TORCH_CHECK(initials.scalar_type() == input.scalar_type(),
                "Initials must have the same scalar type as input");
    TORCH_CHECK(weights.scalar_type() == input.scalar_type(),
                "Weights must have the same scalar type as input");

    auto input_contiguous = input.transpose(0, 1).contiguous();
    auto weights_contiguous = weights.transpose(0, 1).contiguous();
    auto output = at::empty_like(input_contiguous);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        input.scalar_type(), "compute_linear_recurrence", [&] {
            compute_linear_recurrence<scalar_t>(
                weights_contiguous.const_data_ptr<scalar_t>(),
                input_contiguous.const_data_ptr<scalar_t>(),
                initials.const_data_ptr<scalar_t>(),
                output.mutable_data_ptr<scalar_t>(), input_contiguous.size(1),
                input_contiguous.size(0));
        });
    return output.transpose(0, 1).contiguous();
}

TORCH_LIBRARY_IMPL(torchlpc, CUDA, m) { m.impl("scan", &scan_cuda_wrapper); }
