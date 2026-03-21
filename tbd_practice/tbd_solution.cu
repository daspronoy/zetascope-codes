/**
 * Track-Before-Detect: Brute-Force Shift-and-Stack — CUDA Reference Solution
 * ===========================================================================
 *
 * This is the reference CUDA/C++ implementation for evaluating interview
 * candidates. It demonstrates:
 *
 *   1. Efficient 2D spatial parallelization over (x0, y0)
 *   2. Constant memory for broadcast-access velocity grids
 *   3. __ldg() intrinsics for read-only global memory (L1/texture cache)
 *   4. Per-thread local accumulation with register-level tracking
 *   5. Two-stage reduction: per-pixel best → global best
 *   6. CUDA events for accurate kernel timing
 *
 * Compile:
 *   nvcc -O3 -arch=sm_70 -o tbd_solution tbd_solution.cu
 *
 * Run:
 *   ./tbd_solution tbd_frames.bin tbd_params.bin
 *
 * Expected output should match tbd_reference.txt from the Python script.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <cuda_runtime.h>

// ============================================================
// Error checking macro
// ============================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// Constants
// ============================================================

// Velocity grid described by range + count (linspace); supports arbitrary N_VEL
__constant__ float d_v_min;             // Min velocity (pixels/frame)
__constant__ float d_v_max;             // Max velocity (pixels/frame)
__constant__ int   d_n_vel;             // Number of velocity hypotheses per axis
__constant__ int   d_n_frames;          // Number of frames

// Frame dimensions in constant memory
__constant__ int d_nx;
__constant__ int d_ny;

// ============================================================
// Kernel 1: Per-pixel best trajectory search
// ============================================================
// Each thread handles one (x0, y0) starting position and iterates
// over all (vx, vy) velocity hypotheses, tracking the local best.
//
// Memory access pattern:
//   - Velocity grid: constant memory (broadcast to all threads)
//   - Frame data: __ldg() for read-only L1/texture cache path
//   - Output: one write per thread (coalesced by x0)
// ============================================================
__global__ void tbd_search_kernel(
    const float* __restrict__ frames,   // [n_frames, ny, nx] in global memory
    float*  best_stats,                 // [ny, nx] output: best statistic per pixel
    int*    best_vxi,                   // [ny, nx] output: best vx index
    int*    best_vyi                    // [ny, nx] output: best vy index
)
{
    const int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = blockIdx.y * blockDim.y + threadIdx.y;

    if (x0 >= d_nx || y0 >= d_ny) return;

    const int nx = d_nx;
    const int ny = d_ny;
    const int nf = d_n_frames;
    const int nv = d_n_vel;
    const int frame_stride = ny * nx;   // Stride between frames

    float local_best = -FLT_MAX;
    int   local_vxi  = 0;
    int   local_vyi  = 0;

    // Linspace step: v_min + i * (v_max - v_min) / (n_vel - 1)
    const float v_step = (nv > 1) ? (d_v_max - d_v_min) / (float)(nv - 1) : 0.0f;

    // Loop over all velocity hypotheses
    for (int vi = 0; vi < nv; vi++) {
        const float vx = d_v_min + vi * v_step;

        for (int vj = 0; vj < nv; vj++) {
            const float vy = d_v_min + vj * v_step;

            // Shift-and-stack along this trajectory
            float stat = 0.0f;
            bool valid = true;

            #pragma unroll 4
            for (int t = 0; t < nf; t++) {
                // Predicted position at frame t
                int px = __float2int_rn((float)x0 + vx * (float)t);
                int py = __float2int_rn((float)y0 + vy * (float)t);

                if (px < 0 || px >= nx || py < 0 || py >= ny) {
                    valid = false;
                    break;
                }

                // Read through read-only cache (__ldg)
                stat += __ldg(&frames[t * frame_stride + py * nx + px]);
            }

            if (valid && stat > local_best) {
                local_best = stat;
                local_vxi  = vi;
                local_vyi  = vj;
            }
        }
    }

    // Write per-pixel result (coalesced in x)
    const int idx = y0 * nx + x0;
    best_stats[idx] = local_best;
    best_vxi[idx]   = local_vxi;
    best_vyi[idx]   = local_vyi;
}


// ============================================================
// Kernel 2: Parallel reduction to find global maximum
// ============================================================
// Two-pass block reduction + final block reduction.
// Each block finds its local max, then a single block reduces
// across block-level results.
// ============================================================
struct TBDResult {
    float stat;
    int   pixel_idx;
    int   vxi;
    int   vyi;
};

__global__ void reduce_max_kernel(
    const float* __restrict__ stats,
    const int*   __restrict__ vxi,
    const int*   __restrict__ vyi,
    int n_pixels,
    TBDResult* block_results     // [gridDim.x] output
)
{
    extern __shared__ char smem[];
    float* s_stat = (float*)smem;
    int*   s_idx  = (int*)(s_stat + blockDim.x);
    int*   s_vxi  = s_idx + blockDim.x;
    int*   s_vyi  = s_vxi + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load or set sentinel
    if (gid < n_pixels) {
        s_stat[tid] = stats[gid];
        s_idx[tid]  = gid;
        s_vxi[tid]  = vxi[gid];
        s_vyi[tid]  = vyi[gid];
    } else {
        s_stat[tid] = -FLT_MAX;
        s_idx[tid]  = -1;
        s_vxi[tid]  = 0;
        s_vyi[tid]  = 0;
    }
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_stat[tid + stride] > s_stat[tid]) {
            s_stat[tid] = s_stat[tid + stride];
            s_idx[tid]  = s_idx[tid + stride];
            s_vxi[tid]  = s_vxi[tid + stride];
            s_vyi[tid]  = s_vyi[tid + stride];
        }
        __syncthreads();
    }

    // Block winner writes to output
    if (tid == 0) {
        block_results[blockIdx.x].stat      = s_stat[0];
        block_results[blockIdx.x].pixel_idx = s_idx[0];
        block_results[blockIdx.x].vxi       = s_vxi[0];
        block_results[blockIdx.x].vyi       = s_vyi[0];
    }
}


// ============================================================
// Host-side I/O
// ============================================================
float* load_frames(const char* filename, int* nx, int* ny, int* nf) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    int header[3];
    if (fread(header, sizeof(int), 3, f) != 3) {
        fprintf(stderr, "Failed to read header\n"); exit(1);
    }
    *nx = header[0];
    *ny = header[1];
    *nf = header[2];

    size_t n_pixels = (size_t)(*nx) * (*ny) * (*nf);
    float* data = (float*)malloc(n_pixels * sizeof(float));
    if (fread(data, sizeof(float), n_pixels, f) != n_pixels) {
        fprintf(stderr, "Failed to read frame data\n"); exit(1);
    }
    fclose(f);
    return data;
}

void load_params(const char* filename, float* v_min, float* v_max, int* n_vel) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    float params[3];
    if (fread(params, sizeof(float), 3, f) != 3) {
        fprintf(stderr, "Failed to read params\n"); exit(1);
    }
    *v_min = params[0];
    *v_max = params[1];
    float v_step = params[2];
    // Derive n_vel from the step: same formula as np.arange in Python
    *n_vel = (int)roundf((*v_max - *v_min) / v_step) + 1;
    fclose(f);
}


// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    const char* frame_file = (argc > 1) ? argv[1] : "tbd_frames.bin";
    const char* param_file = (argc > 2) ? argv[2] : "tbd_params.bin";

    printf("================================================================\n");
    printf("  Track-Before-Detect: CUDA Reference Solution\n");
    printf("================================================================\n\n");

    // --- Print GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %d SMs, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    // --- Start total wall-clock timer ---
    auto total_start = std::chrono::steady_clock::now();

    // --- Load data ---
    int nx, ny, nf;
    float* h_frames = load_frames(frame_file, &nx, &ny, &nf);
    printf("Loaded frames: %d x %d x %d frames\n", nx, ny, nf);

    float v_min, v_max;
    int n_vel;
    load_params(param_file, &v_min, &v_max, &n_vel);

    int n_pixels = nx * ny;
    long long total_traj = (long long)n_pixels * n_vel * n_vel;
    long long total_reads = total_traj * nf;

    printf("Velocity grid : %d x %d = %d hypotheses  (%.2f to %.2f)\n",
           n_vel, n_vel, n_vel * n_vel, v_min, v_max);
    printf("Total trajectories : %lld\n", total_traj);
    printf("Total pixel reads  : %lld\n", total_reads);
    printf("\n");

    // --- Copy constants to device ---
    CUDA_CHECK(cudaMemcpyToSymbol(d_v_min,    &v_min, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_v_max,    &v_max, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_vel,    &n_vel, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_frames, &nf,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nx,       &nx,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ny,       &ny,    sizeof(int)));

    // --- Allocate device memory ---
    size_t frame_bytes = (size_t)nx * ny * nf * sizeof(float);
    size_t pixel_bytes = (size_t)n_pixels * sizeof(float);
    size_t idx_bytes   = (size_t)n_pixels * sizeof(int);

    float* d_frames;
    float* d_best_stats;
    int*   d_best_vxi;
    int*   d_best_vyi;

    CUDA_CHECK(cudaMalloc(&d_frames,     frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_stats, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vxi,   idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vyi,   idx_bytes));

    // --- Copy frames to device ---
    CUDA_CHECK(cudaMemcpy(d_frames, h_frames, frame_bytes, cudaMemcpyHostToDevice));
    printf("Device memory allocated: %.1f MB\n",
           (frame_bytes + pixel_bytes + 2 * idx_bytes) / 1e6);

    // --- Configure kernel launch ---
    dim3 block(16, 16);     // 256 threads per block
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    printf("Kernel config : grid(%d, %d) x block(%d, %d) = %d threads\n",
           grid.x, grid.y, block.x, block.y,
           grid.x * grid.y * block.x * block.y);
    printf("\n");

    // --- Launch search kernel with timing ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    tbd_search_kernel<<<grid, block>>>(d_frames, d_best_stats, d_best_vxi, d_best_vyi);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));

    tbd_search_kernel<<<grid, block>>>(d_frames, d_best_stats, d_best_vxi, d_best_vyi);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    printf("Search kernel time: %.3f ms\n", kernel_ms);

    // --- Reduction to find global maximum ---
    CUDA_CHECK(cudaEventRecord(start));

    const int RED_BLOCK = 256;
    int red_grid = (n_pixels + RED_BLOCK - 1) / RED_BLOCK;
    size_t smem_size = RED_BLOCK * (sizeof(float) + 3 * sizeof(int));

    TBDResult* d_block_results;
    CUDA_CHECK(cudaMalloc(&d_block_results, red_grid * sizeof(TBDResult)));

    reduce_max_kernel<<<red_grid, RED_BLOCK, smem_size>>>(
        d_best_stats, d_best_vxi, d_best_vyi, n_pixels, d_block_results);

    // Copy block results to host for final reduction
    TBDResult* h_block_results = (TBDResult*)malloc(red_grid * sizeof(TBDResult));
    CUDA_CHECK(cudaMemcpy(h_block_results, d_block_results,
                          red_grid * sizeof(TBDResult), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float reduce_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&reduce_ms, start, stop));

    // Final reduction on host (over block results)
    TBDResult best = {-FLT_MAX, -1, 0, 0};
    for (int i = 0; i < red_grid; i++) {
        if (h_block_results[i].stat > best.stat) {
            best = h_block_results[i];
        }
    }

    printf("Reduction time    : %.3f ms\n", reduce_ms);
    printf("Total GPU time    : %.3f ms\n", kernel_ms + reduce_ms);
    printf("\n");

    // --- Decode result ---
    int det_x0 = best.pixel_idx % nx;
    int det_y0 = best.pixel_idx / nx;
    float v_step_out = (n_vel > 1) ? (v_max - v_min) / (float)(n_vel - 1) : 0.0f;
    float det_vx = v_min + best.vxi * v_step_out;
    float det_vy = v_min + best.vyi * v_step_out;

    printf("================================================================\n");
    printf("  RESULT\n");
    printf("================================================================\n");
    printf("  Detected : x0=%d, y0=%d, vx=%.2f, vy=%.2f\n",
           det_x0, det_y0, det_vx, det_vy);
    printf("  Statistic: %.1f\n", best.stat);
    printf("  Kernel throughput: %.2f G trajectories/s\n",
           total_traj / (kernel_ms * 1e6));
    printf("  Effective bandwidth: %.2f GB/s (pixel reads)\n",
           total_reads * sizeof(float) / (kernel_ms * 1e6));

    auto total_stop = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    printf("  Total GPU time    : %.3f ms\n", kernel_ms + reduce_ms);
    printf("  Total wall time   : %.3f ms\n", total_ms);
    printf("================================================================\n");

    // --- Cleanup ---
    free(h_frames);
    free(h_block_results);
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_best_stats));
    CUDA_CHECK(cudaFree(d_best_vxi));
    CUDA_CHECK(cudaFree(d_best_vyi));
    CUDA_CHECK(cudaFree(d_block_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
