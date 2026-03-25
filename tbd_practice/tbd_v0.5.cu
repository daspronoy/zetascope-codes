/**
 * Orin-Optimized TBD: Exhaustive Shift-and-Stack with FP16 Frames
 * ================================================================
 *
 * Targets: NVIDIA Jetson AGX Orin (sm_87, Ampere, 2048 CUDA cores,
 *          ~200 GB/s unified memory bandwidth, 32-64 GB unified RAM)
 *
 * Optimizations over the reference solution (tbd_solution.cu):
 *
 *   1. FP16 FRAME STORAGE (halves memory footprint & bandwidth)
 *      Frames are stored as __half on device; each trajectory read
 *      converts to FP32 for accumulation. Photon counts ~100 are
 *      well within FP16 range (max representable ~65504).
 *
 *   2. ORIN-SPECIFIC THREAD BLOCK TUNING
 *      8x16 = 128 threads per block, tuned for 2048 CUDA cores / 16 SMs
 *      on sm_87. Balances occupancy against register pressure.
 *
 *   3. ROW-CHUNKED KERNEL LAUNCHES
 *      Splits the NY rows into fixed-size chunks to avoid GPU watchdog
 *      timeouts on long-running kernels and to enable progress reporting.
 *
 *   4. GPU-SIDE ARGMAX REDUCTION
 *      The per-pixel best-score array is reduced entirely on the GPU.
 *      Only ~10K block-level results are copied to host, avoiding a
 *      large device-to-host transfer for the full n_pixels array.
 *
 * NOTE on coarse-to-fine pyramids:
 *   A velocity pyramid (coarse-to-fine) is NOT suitable for this problem.
 *   The shift-and-stack SNR at the coarse level is proportional to
 *   (0.5 / coarse_step) * signal / sqrt(bg * n_frames). For n_frames=16
 *   and coarse_step=0.50 this gives SNR~1.5 sigma — indistinguishable
 *   from noise — so the correct velocity is not reliably retained at the
 *   coarse level. Pyramids only help when n_frames is large enough that
 *   even a misaligned coarse stack yields a clear signal.
 *
 * Compile (Orin):
 *   nvcc -O3 -arch=sm_87 --use_fast_math -o tbd_orin tbd_orin_optimized.cu
 *
 * Compile (dev, e.g. RTX 4090):
 *   nvcc -O3 -arch=sm_89 --use_fast_math -o tbd_orin tbd_orin_optimized.cu
 *
 * Run:
 *   ./tbd_orin tbd_frames.bin tbd_params.bin
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================
// Error checking
// ============================================================
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Row chunking for kernel launches (avoids watchdog timeout)
#define CHUNK_ROWS  128

// ============================================================
// Constant memory
// ============================================================
__constant__ int   d_n_frames;
__constant__ int   d_nx;
__constant__ int   d_ny;
__constant__ float d_v_min;
__constant__ float d_v_max;
__constant__ int   d_n_vel;

// ============================================================
// Kernel: Exhaustive shift-and-stack search (FP16 frames)
// ============================================================
// Each thread handles one (x0, y0) starting pixel and iterates
// over the full (vx, vy) velocity grid. Frames are read as FP16
// and accumulated in FP32. Best (stat, vx, vy) stored per pixel.
// ============================================================
__global__ void tbd_search(
    const __half* __restrict__ frames,
    float* best_stats,
    float* best_vx_out,
    float* best_vy_out,
    int    y_start,
    int    y_end)
{
    const int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = y_start + blockIdx.y * blockDim.y + threadIdx.y;

    if (x0 >= d_nx || y0 >= y_end) return;

    const int nx = d_nx;
    const int ny = d_ny;
    const int nf = d_n_frames;
    const int nv = d_n_vel;
    const int frame_stride = ny * nx;

    const float v_step = (nv > 1) ? (d_v_max - d_v_min) / (float)(nv - 1) : 0.0f;

    float local_best = -FLT_MAX;
    float local_vx   = 0.0f;
    float local_vy   = 0.0f;

    for (int vi = 0; vi < nv; vi++) {
        const float vx = d_v_min + vi * v_step;

        for (int vj = 0; vj < nv; vj++) {
            const float vy = d_v_min + vj * v_step;

            float stat  = 0.0f;
            bool  valid = true;

            #pragma unroll 4
            for (int t = 0; t < nf; t++) {
                int px = __float2int_rn((float)x0 + vx * (float)t);
                int py = __float2int_rn((float)y0 + vy * (float)t);

                if (px < 0 || px >= nx || py < 0 || py >= ny) {
                    valid = false;
                    break;
                }

                stat += __half2float(
                    __ldg(&frames[(size_t)t * frame_stride + py * nx + px]));
            }

            if (valid && stat > local_best) {
                local_best = stat;
                local_vx   = vx;
                local_vy   = vy;
            }
        }
    }

    const int idx = y0 * nx + x0;
    best_stats[idx]  = local_best;
    best_vx_out[idx] = local_vx;
    best_vy_out[idx] = local_vy;
}

// ============================================================
// Kernel: GPU-side argmax reduction
// ============================================================
__global__ void reduce_argmax(
    const float* __restrict__ stats,
    const float* __restrict__ vx_arr,
    const float* __restrict__ vy_arr,
    int    n,
    float* block_stats,
    float* block_vx,
    float* block_vy,
    int*   block_idx)
{
    extern __shared__ char smem[];
    float* s_stats = (float*)smem;
    int*   s_idx   = (int*)(s_stats + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        s_stats[tid] = stats[gid];
        s_idx[tid]   = gid;
    } else {
        s_stats[tid] = -FLT_MAX;
        s_idx[tid]   = -1;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_stats[tid + stride] > s_stats[tid]) {
            s_stats[tid] = s_stats[tid + stride];
            s_idx[tid]   = s_idx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_stats[blockIdx.x] = s_stats[0];
        block_idx[blockIdx.x]   = s_idx[0];
        int best = s_idx[0];
        block_vx[blockIdx.x] = (best >= 0 && best < n) ? vx_arr[best] : 0.0f;
        block_vy[blockIdx.x] = (best >= 0 && best < n) ? vy_arr[best] : 0.0f;
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
    printf("  Orin-Optimized TBD: Exhaustive Search with FP16 Frames\n");
    printf("================================================================\n\n");

    // --- GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB memory)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    printf("L2 cache: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("Unified memory: %s\n\n", prop.unifiedAddressing ? "yes" : "no");

    // --- Start total wall-clock timer ---
    auto total_start = std::chrono::steady_clock::now();

    // --- Load data from disk ---
    printf("[Phase 1] Loading data from disk...\n");
    auto load_t0 = std::chrono::steady_clock::now();

    int nx, ny, nf;
    float* h_frames_fp32 = load_frames(frame_file, &nx, &ny, &nf);
    printf("  Frames : %d x %d x %d\n", nx, ny, nf);

    float v_min, v_max;
    int n_vel;
    load_params(param_file, &v_min, &v_max, &n_vel);
    printf("  Velocity grid : %d hypotheses per axis  (%.4f to %.4f)\n",
           n_vel, v_min, v_max);

    auto load_t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
    printf("  Load time : %.3f ms\n\n", load_ms);

    int n_pixels = nx * ny;
    long long total_traj  = (long long)n_pixels * n_vel * n_vel;
    long long total_reads = total_traj * nf;

    printf("Problem size:\n");
    printf("  Image     : %d x %d = %d pixels\n", nx, ny, n_pixels);
    printf("  Frames    : %d\n", nf);
    printf("  Velocities: %d x %d = %d hypotheses/pixel\n", n_vel, n_vel, n_vel * n_vel);
    printf("  Total traj: %.3f billion\n", total_traj / 1e9);
    printf("  FP16 frame data: %.1f MB  (vs %.1f MB FP32)\n",
           (double)n_pixels * nf * sizeof(__half) / 1e6,
           (double)n_pixels * nf * sizeof(float) / 1e6);
    printf("\n");

    // --- Copy constants to GPU ---
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_frames, &nf,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nx,       &nx,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ny,       &ny,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_v_min,    &v_min, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_v_max,    &v_max, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_vel,    &n_vel, sizeof(int)));

    // --- Allocate device memory ---
    size_t frame_bytes = (size_t)n_pixels * nf * sizeof(__half);
    size_t stat_bytes  = (size_t)n_pixels * sizeof(float);

    printf("Allocating device memory (%.1f MB frames + %.1f MB outputs)...\n",
           frame_bytes / 1e6, 3 * stat_bytes / 1e6);

    __half* d_frames;
    float*  d_best_stats;
    float*  d_best_vx;
    float*  d_best_vy;

    CUDA_CHECK(cudaMalloc(&d_frames,     frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_stats, stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vx,    stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vy,    stat_bytes));

    // --- Convert FP32 -> FP16 and upload ---
    printf("Converting FP32 -> FP16 and uploading frames...\n");
    auto up_t0 = std::chrono::steady_clock::now();

    size_t n_total = (size_t)n_pixels * nf;
    __half* h_frames_fp16 = (__half*)malloc(frame_bytes);
    for (size_t i = 0; i < n_total; i++)
        h_frames_fp16[i] = __float2half(h_frames_fp32[i]);

    free(h_frames_fp32);

    CUDA_CHECK(cudaMemcpy(d_frames, h_frames_fp16, frame_bytes, cudaMemcpyHostToDevice));
    free(h_frames_fp16);

    auto up_t1 = std::chrono::steady_clock::now();
    double up_ms = std::chrono::duration<double, std::milli>(up_t1 - up_t0).count();
    printf("  Done (%.3f ms)\n\n", up_ms);

    // === PHASE 2: Exhaustive TBD search ===
    printf("[Phase 2] Running exhaustive TBD search (FP16 frames)...\n");
    printf("  Launching in chunks of %d rows\n\n", CHUNK_ROWS);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // 8x16 = 128 threads/block, tuned for Orin's 2048 cores / 16 SMs
    dim3 block(8, 16);
    float total_search_ms = 0.0f;
    int n_chunks = (ny + CHUNK_ROWS - 1) / CHUNK_ROWS;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int y_start    = chunk * CHUNK_ROWS;
        int y_end      = y_start + CHUNK_ROWS;
        if (y_end > ny) y_end = ny;
        int chunk_rows = y_end - y_start;

        dim3 grid((nx + block.x - 1) / block.x,
                  (chunk_rows + block.y - 1) / block.y);

        CUDA_CHECK(cudaEventRecord(t0));

        tbd_search<<<grid, block>>>(
            d_frames, d_best_stats, d_best_vx, d_best_vy,
            y_start, y_end);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float chunk_ms;
        CUDA_CHECK(cudaEventElapsedTime(&chunk_ms, t0, t1));
        total_search_ms += chunk_ms;

        float pct          = 100.0f * y_end / ny;
        float rows_per_sec = y_end / (total_search_ms * 0.001f);
        float eta_s        = (ny - y_end) / rows_per_sec;

        printf("  Rows %4d-%4d / %d  (%5.1f%%)  "
               "chunk %.1fms  total %.1fms  ETA %.0fs\n",
               y_start, y_end, ny, pct,
               chunk_ms, total_search_ms, eta_s);
        fflush(stdout);
    }

    printf("\n  Search complete. Total kernel time: %.3f ms\n\n",
           total_search_ms);

    // === PHASE 3: GPU-side argmax reduction ===
    printf("[Phase 3] GPU-side argmax reduction...\n");

    int    red_block = 1024;
    int    red_grid  = (n_pixels + red_block - 1) / red_block;
    size_t smem_size = red_block * (sizeof(float) + sizeof(int));

    float* d_blk_stats;
    float* d_blk_vx;
    float* d_blk_vy;
    int*   d_blk_idx;

    CUDA_CHECK(cudaMalloc(&d_blk_stats, red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vx,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vy,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_idx,   red_grid * sizeof(int)));

    reduce_argmax<<<red_grid, red_block, smem_size>>>(
        d_best_stats, d_best_vx, d_best_vy,
        n_pixels,
        d_blk_stats, d_blk_vx, d_blk_vy, d_blk_idx);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_blk_stats = (float*)malloc(red_grid * sizeof(float));
    float* h_blk_vx    = (float*)malloc(red_grid * sizeof(float));
    float* h_blk_vy    = (float*)malloc(red_grid * sizeof(float));
    int*   h_blk_idx   = (int*)  malloc(red_grid * sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_blk_stats, d_blk_stats, red_grid * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_vx,    d_blk_vx,    red_grid * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_vy,    d_blk_vy,    red_grid * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_idx,   d_blk_idx,   red_grid * sizeof(int),   cudaMemcpyDeviceToHost));

    float best_stat = -FLT_MAX;
    int   best_idx  = -1;
    float det_vx    = 0.0f;
    float det_vy    = 0.0f;

    for (int i = 0; i < red_grid; i++) {
        if (h_blk_stats[i] > best_stat) {
            best_stat = h_blk_stats[i];
            best_idx  = h_blk_idx[i];
            det_vx    = h_blk_vx[i];
            det_vy    = h_blk_vy[i];
        }
    }

    int det_x0 = best_idx % nx;
    int det_y0 = best_idx / nx;

    // === RESULTS ===
    auto total_stop    = std::chrono::steady_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(
                               total_stop - total_start).count();

    printf("\n");
    printf("================================================================\n");
    printf("  RESULTS\n");
    printf("================================================================\n");
    printf("  Detected trajectory:\n");
    printf("    x0   = %d\n",            det_x0);
    printf("    y0   = %d\n",            det_y0);
    printf("    vx   = %.4f px/frame\n", det_vx);
    printf("    vy   = %.4f px/frame\n", det_vy);
    printf("    stat = %.1f\n",          best_stat);
    printf("\n");
    printf("  Timing:\n");
    printf("    Data load + convert : %.3f ms\n", load_ms + up_ms);
    printf("    TBD search          : %.3f ms\n", total_search_ms);
    printf("    Total wall time     : %.3f ms\n", total_wall_ms);
    printf("\n");
    printf("  Throughput:\n");
    printf("    %.3f billion trajectories / s\n",
           total_traj / (total_search_ms * 1e6));
    printf("    %.3f trillion pixel reads / s\n",
           total_reads / (total_search_ms * 1e9));
    printf("    Effective BW (FP16 reads): %.2f GB/s\n",
           total_reads * sizeof(__half) / (total_search_ms * 1e6));
    printf("================================================================\n");

    // --- Cleanup ---
    free(h_blk_stats); free(h_blk_vx); free(h_blk_vy); free(h_blk_idx);
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_best_stats));
    CUDA_CHECK(cudaFree(d_best_vx));
    CUDA_CHECK(cudaFree(d_best_vy));
    CUDA_CHECK(cudaFree(d_blk_stats));
    CUDA_CHECK(cudaFree(d_blk_vx));
    CUDA_CHECK(cudaFree(d_blk_vy));
    CUDA_CHECK(cudaFree(d_blk_idx));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    return 0;
}
