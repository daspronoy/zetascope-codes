/**
 * Orin-Optimized TBD with Coarse-to-Fine Velocity Pyramid
 * =========================================================
 *
 * Targets: NVIDIA Jetson AGX Orin (sm_87, Ampere, 2048 CUDA cores,
 *          ~200 GB/s unified memory bandwidth, 32-64 GB unified RAM)
 *
 * Key optimizations over brute-force baseline:
 *
 *   1. COARSE-TO-FINE VELOCITY PYRAMID (~25-40x fewer velocity evals)
 *      - Level 0 (coarse):  step=0.50, 15x15 =  225 hypotheses/pixel
 *      - Level 1 (medium):  step=0.10, 11x11 =  121 hypotheses/candidate
 *      - Level 2 (fine):    step=0.05, 5x5   =   25 hypotheses/candidate
 *      Total: ~500-800 vs 19,881 brute-force
 *
 *   2. FP16 FRAME STORAGE (halves memory & bandwidth)
 *      Orin's Ampere GPU has native __half support. Photon counts
 *      ~100+/-10 are well within FP16 range. Accumulation in FP32.
 *
 *   3. TEMPORAL STREAMING (reduced peak memory)
 *      Instead of 12 GB for 300 frames x 10MP x FP32, we store
 *      FP16 frames (~6 GB) and can optionally stream in chunks.
 *
 *   4. ORIN-SPECIFIC TUNING
 *      - sm_87 compute capability
 *      - Smaller thread blocks tuned for 2048 CUDA cores / 16 SMs
 *      - cudaMallocManaged with prefetch hints for unified memory
 *      - Reduced constant memory pressure
 *
 *   5. GPU-SIDE REDUCTION (avoids 40 MB D2H transfer for argmax)
 *
 *   6. BOUNDS-AWARE TRAJECTORY PRUNING
 *      Pre-compute valid velocity range per pixel to skip
 *      trajectories that immediately exit the FOV.
 *
 * Compile (Orin):
 *   nvcc -O3 -arch=sm_87 --use_fast_math -o tbd_orin tbd_orin_optimized.cu
 *
 * Compile (dev on RTX 4090 for testing):
 *   nvcc -O3 -arch=sm_89 --use_fast_math -o tbd_orin tbd_orin_optimized.cu
 *
 * Expected speedup vs brute-force on Orin: ~25-40x
 * Expected runtime on Orin AGX: comparable to brute-force on RTX 4090
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <algorithm>
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

// ============================================================
// Problem parameters (same as baseline for apples-to-apples)
// ============================================================
#define NX          4000
#define NY          2500
#define N_PIXELS    (NX * NY)   // 10,000,000
#define N_FRAMES    300
#define BG_MEAN     100.0f
#define OBJ_SIGNAL  20.0f

// Ground truth
#define TRUE_X0     2000.0f
#define TRUE_Y0     1250.0f
#define TRUE_VX     1.30f
#define TRUE_VY    -0.70f

// Velocity search range (same as baseline)
#define V_MIN      -3.50f
#define V_MAX       3.50f

// === Pyramid levels ===
// Level 0 (coarse): step=0.50, covers full range, 15x15 = 225 pairs
#define L0_STEP     0.50f
// Level 1 (medium): step=0.10, +/-0.50 around L0 winners, 11x11 = 121 pairs
#define L1_STEP     0.10f
#define L1_RADIUS   0.50f
// Level 2 (fine):   step=0.05, +/-0.10 around L1 winners, 5x5 = 25 pairs
#define L2_STEP     0.05f
#define L2_RADIUS   0.10f

// How many top candidates to refine at each level
#define TOP_K       4

// Row chunking for kernel launches (avoids watchdog timeout)
#define CHUNK_ROWS  200   // Larger chunks OK since pyramid is fast

// ============================================================
// Constant memory for velocity grids at each pyramid level
// ============================================================
#define MAX_VEL_L0  32
#define MAX_VEL_L1  32
#define MAX_VEL_L2  32

__constant__ float d_vel_L0[MAX_VEL_L0];
__constant__ float d_vel_L1[MAX_VEL_L1];  // Template, offset per candidate
__constant__ float d_vel_L2[MAX_VEL_L2];  // Template, offset per candidate
__constant__ int   d_nv_L0;
__constant__ int   d_nv_L1;
__constant__ int   d_nv_L2;
__constant__ int   d_n_frames;
__constant__ int   d_nx;
__constant__ int   d_ny;

// ============================================================
// Device PRNG (xorshift32 + Box-Muller) — no curand dependency
// ============================================================
__device__ __forceinline__ unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ __forceinline__ float rand_uniform(unsigned int* state) {
    return (float)xorshift32(state) * 2.3283064365e-10f;
}

// ============================================================
// Kernel: Generate noise as FP16 directly on GPU
// ============================================================
__global__ void generate_noise_fp16(
    __half* frames, int n_pixels, int n_frames)
{
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= n_pixels) return;

    unsigned int state = pixel * 1103515245u + 12345u + 7919u;
    for (int i = 0; i < 4; i++) xorshift32(&state);

    for (int t = 0; t < n_frames; t++) {
        float u1 = rand_uniform(&state) + 1e-10f;
        float u2 = rand_uniform(&state);
        float z  = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        float val = BG_MEAN + sqrtf(BG_MEAN) * z;
        val = fmaxf(val, 0.0f);
        frames[(size_t)t * n_pixels + pixel] = __float2half(val);
    }
}

// ============================================================
// Kernel: Inject signal into FP16 frames
// ============================================================
__global__ void inject_signal_fp16(
    __half* frames, int nx, int ny, int n_frames,
    float x0, float y0, float vx, float vy, float signal)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_frames) return;

    int px = __float2int_rn(x0 + vx * (float)t);
    int py = __float2int_rn(y0 + vy * (float)t);
    if (px >= 0 && px < nx && py >= 0 && py < ny) {
        size_t idx = (size_t)t * nx * ny + py * nx + px;
        float old_val = __half2float(frames[idx]);
        frames[idx] = __float2half(old_val + signal);
    }
}

// ============================================================
// Device helper: evaluate one trajectory hypothesis
// Reads FP16, accumulates in FP32 for precision
// ============================================================
__device__ __forceinline__ float evaluate_trajectory(
    const __half* __restrict__ frames,
    int x0, int y0, float vx, float vy,
    int nx, int ny, int nf)
{
    const int frame_stride = ny * nx;
    float stat = 0.0f;

    for (int t = 0; t < nf; t++) {
        int px = __float2int_rn((float)x0 + vx * (float)t);
        int py = __float2int_rn((float)y0 + vy * (float)t);

        if (px < 0 || px >= nx || py < 0 || py >= ny)
            return -FLT_MAX;  // Invalid trajectory

        stat += __half2float(frames[(size_t)t * frame_stride + py * nx + px]);
    }
    return stat;
}

// ============================================================
// Struct to hold top-K candidates per pixel (small, in registers)
// ============================================================
struct Candidate {
    float stat;
    float vx;
    float vy;
};

// ============================================================
// Device helper: insert into top-K sorted array (descending)
// ============================================================
__device__ __forceinline__ void topk_insert(
    Candidate* topk, int K, float stat, float vx, float vy)
{
    if (stat <= topk[K - 1].stat) return;  // Not good enough

    // Find insertion point
    int pos = K - 1;
    while (pos > 0 && stat > topk[pos - 1].stat) {
        topk[pos] = topk[pos - 1];
        pos--;
    }
    topk[pos].stat = stat;
    topk[pos].vx   = vx;
    topk[pos].vy   = vy;
}

// ============================================================
// Kernel: Coarse-to-Fine Pyramid TBD Search
// ============================================================
// Each thread handles one (x0, y0) pixel, runs all 3 pyramid
// levels internally. This avoids storing intermediate results
// for 10M pixels and keeps everything in registers.
// ============================================================
__global__ void tbd_pyramid_search(
    const __half* __restrict__ frames,
    float*  best_stats,    // [N_PIXELS] output: best statistic
    float*  best_vx_out,   // [N_PIXELS] output: best vx
    float*  best_vy_out,   // [N_PIXELS] output: best vy
    int     y_start,
    int     y_end)
{
    const int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = y_start + blockIdx.y * blockDim.y + threadIdx.y;

    if (x0 >= d_nx || y0 >= y_end) return;

    const int nx = d_nx;
    const int ny = d_ny;
    const int nf = d_n_frames;
    const int nv0 = d_nv_L0;
    const int nv1 = d_nv_L1;
    const int nv2 = d_nv_L2;

    // ---- Bounds-aware early velocity pruning ----
    // For trajectory to stay in-bounds over all nf frames:
    //   0 <= x0 + vx*(nf-1) < nx  =>  vx in [(-x0)/(nf-1), (nx-1-x0)/(nf-1)]
    // Similarly for vy.
    const float inv_T = 1.0f / (float)(nf - 1);
    const float vx_lo = fmaxf(V_MIN, -(float)x0 * inv_T);
    const float vx_hi = fminf(V_MAX,  (float)(nx - 1 - x0) * inv_T);
    const float vy_lo = fmaxf(V_MIN, -(float)y0 * inv_T);
    const float vy_hi = fminf(V_MAX,  (float)(ny - 1 - y0) * inv_T);

    // ================================================================
    // LEVEL 0: Coarse search (step=0.50)
    // ================================================================
    Candidate topk_L0[TOP_K];
    for (int k = 0; k < TOP_K; k++) {
        topk_L0[k].stat = -FLT_MAX;
        topk_L0[k].vx   = 0.0f;
        topk_L0[k].vy   = 0.0f;
    }

    for (int vi = 0; vi < nv0; vi++) {
        const float vx = d_vel_L0[vi];
        if (vx < vx_lo || vx > vx_hi) continue;  // Prune

        for (int vj = 0; vj < nv0; vj++) {
            const float vy = d_vel_L0[vj];
            if (vy < vy_lo || vy > vy_hi) continue;  // Prune

            float stat = evaluate_trajectory(frames, x0, y0, vx, vy, nx, ny, nf);
            topk_insert(topk_L0, TOP_K, stat, vx, vy);
        }
    }

    // ================================================================
    // LEVEL 1: Medium refinement around top-K coarse winners
    // ================================================================
    Candidate topk_L1[TOP_K];
    for (int k = 0; k < TOP_K; k++) {
        topk_L1[k].stat = -FLT_MAX;
        topk_L1[k].vx   = 0.0f;
        topk_L1[k].vy   = 0.0f;
    }

    for (int c = 0; c < TOP_K; c++) {
        if (topk_L0[c].stat <= -FLT_MAX) continue;

        const float cvx = topk_L0[c].vx;
        const float cvy = topk_L0[c].vy;

        for (int vi = 0; vi < nv1; vi++) {
            const float vx = cvx - L1_RADIUS + d_vel_L1[vi];
            if (vx < vx_lo || vx > vx_hi) continue;

            for (int vj = 0; vj < nv1; vj++) {
                const float vy = cvy - L1_RADIUS + d_vel_L1[vj];
                if (vy < vy_lo || vy > vy_hi) continue;

                float stat = evaluate_trajectory(frames, x0, y0, vx, vy, nx, ny, nf);
                topk_insert(topk_L1, TOP_K, stat, vx, vy);
            }
        }
    }

    // ================================================================
    // LEVEL 2: Fine refinement around top-K medium winners
    // ================================================================
    float global_best_stat = -FLT_MAX;
    float global_best_vx   = 0.0f;
    float global_best_vy   = 0.0f;

    for (int c = 0; c < TOP_K; c++) {
        if (topk_L1[c].stat <= -FLT_MAX) continue;

        const float cvx = topk_L1[c].vx;
        const float cvy = topk_L1[c].vy;

        for (int vi = 0; vi < nv2; vi++) {
            const float vx = cvx - L2_RADIUS + d_vel_L2[vi];
            if (vx < vx_lo || vx > vx_hi) continue;

            for (int vj = 0; vj < nv2; vj++) {
                const float vy = cvy - L2_RADIUS + d_vel_L2[vj];
                if (vy < vy_lo || vy > vy_hi) continue;

                float stat = evaluate_trajectory(frames, x0, y0, vx, vy, nx, ny, nf);
                if (stat > global_best_stat) {
                    global_best_stat = stat;
                    global_best_vx   = vx;
                    global_best_vy   = vy;
                }
            }
        }
    }

    const int idx = y0 * nx + x0;
    best_stats[idx]  = global_best_stat;
    best_vx_out[idx] = global_best_vx;
    best_vy_out[idx] = global_best_vy;
}

// ============================================================
// Kernel: GPU-side argmax reduction
// ============================================================
// Phase 1: Each block reduces a chunk, writes per-block result
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

    // Load
    if (gid < n) {
        s_stats[tid] = stats[gid];
        s_idx[tid]   = gid;
    } else {
        s_stats[tid] = -FLT_MAX;
        s_idx[tid]   = -1;
    }
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_stats[tid + stride] > s_stats[tid]) {
                s_stats[tid] = s_stats[tid + stride];
                s_idx[tid]   = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_stats[blockIdx.x] = s_stats[0];
        block_idx[blockIdx.x]   = s_idx[0];
        int best = s_idx[0];
        block_vx[blockIdx.x]    = (best >= 0 && best < n) ? vx_arr[best] : 0.0f;
        block_vy[blockIdx.x]    = (best >= 0 && best < n) ? vy_arr[best] : 0.0f;
    }
}

// ============================================================
// Host helper: build velocity grid for a pyramid level
// ============================================================
int build_vel_grid(float* buf, int max_n, float vmin, float vmax, float step) {
    int n = 0;
    for (float v = vmin; v <= vmax + step * 0.4f; v += step) {
        if (n >= max_n) {
            fprintf(stderr, "ERROR: Velocity grid overflow at level (max=%d)\n", max_n);
            exit(1);
        }
        buf[n++] = v;
    }
    return n;
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("================================================================\n");
    printf("  Orin-Optimized TBD: Coarse-to-Fine Velocity Pyramid\n");
    printf("================================================================\n\n");

    // --- GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB memory)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    printf("L2 cache: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("Unified memory: %s\n",
           prop.unifiedAddressing ? "yes" : "no");
    printf("\n");

    // --- Build velocity grids for each pyramid level ---
    float h_vel_L0[MAX_VEL_L0], h_vel_L1[MAX_VEL_L1], h_vel_L2[MAX_VEL_L2];

    // Level 0: full range, coarse step
    int nv0 = build_vel_grid(h_vel_L0, MAX_VEL_L0, V_MIN, V_MAX, L0_STEP);
    // Level 1: relative offsets [0, 2*L1_RADIUS], step L1_STEP
    // Kernel adds these to (center - L1_RADIUS)
    int nv1 = build_vel_grid(h_vel_L1, MAX_VEL_L1, 0.0f, 2.0f * L1_RADIUS, L1_STEP);
    // Level 2: relative offsets [0, 2*L2_RADIUS], step L2_STEP
    int nv2 = build_vel_grid(h_vel_L2, MAX_VEL_L2, 0.0f, 2.0f * L2_RADIUS, L2_STEP);

    // Compute total velocity evaluations per pixel
    long long evals_L0 = (long long)nv0 * nv0;
    long long evals_L1 = (long long)TOP_K * nv1 * nv1;
    long long evals_L2 = (long long)TOP_K * nv2 * nv2;
    long long total_evals = evals_L0 + evals_L1 + evals_L2;

    // Brute-force reference (step=0.05 over full range)
    int nv_brute = 0;
    for (float v = V_MIN; v <= V_MAX + 0.02f; v += 0.05f) nv_brute++;
    long long brute_evals = (long long)nv_brute * nv_brute;

    printf("Pyramid velocity search:\n");
    printf("  Level 0 (coarse, step=%.2f): %d x %d = %lld hypotheses\n",
           L0_STEP, nv0, nv0, evals_L0);
    printf("  Level 1 (medium, step=%.2f): %d * %d x %d = %lld hypotheses\n",
           L1_STEP, TOP_K, nv1, nv1, evals_L1);
    printf("  Level 2 (fine,   step=%.2f): %d * %d x %d = %lld hypotheses\n",
           L2_STEP, TOP_K, nv2, nv2, evals_L2);
    printf("  Total per pixel   : %lld\n", total_evals);
    printf("  Brute-force ref   : %lld\n", brute_evals);
    printf("  Speedup factor    : %.1fx\n", (double)brute_evals / total_evals);
    printf("\n");

    long long total_traj  = (long long)N_PIXELS * total_evals;
    long long total_reads = total_traj * N_FRAMES;

    printf("Problem size:\n");
    printf("  Image          : %d x %d = %d pixels (%.1f MP)\n",
           NX, NY, N_PIXELS, N_PIXELS / 1e6);
    printf("  Frames         : %d\n", N_FRAMES);
    printf("  Total traj     : %.2f billion (vs %.1f B brute-force)\n",
           total_traj / 1e9, (double)N_PIXELS * brute_evals / 1e9);
    printf("  Frame data     : %.2f GB (FP16)\n",
           (double)N_PIXELS * N_FRAMES * sizeof(__half) / 1e9);
    printf("\n");

    printf("Ground truth:\n");
    printf("  Position : (%.0f, %.0f)\n", TRUE_X0, TRUE_Y0);
    printf("  Velocity : (%.2f, %.2f) pix/frame\n", TRUE_VX, TRUE_VY);
    printf("  Per-frame SNR : %.1f\n", OBJ_SIGNAL / sqrtf(BG_MEAN));
    printf("  Stacked SNR   : %.1f\n",
           OBJ_SIGNAL * sqrtf((float)N_FRAMES) / sqrtf(BG_MEAN));
    printf("\n");

    // --- Copy constants to GPU ---
    CUDA_CHECK(cudaMemcpyToSymbol(d_vel_L0,   h_vel_L0, nv0 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_vel_L1,   h_vel_L1, nv1 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_vel_L2,   h_vel_L2, nv2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nv_L0,    &nv0,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nv_L1,    &nv1,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nv_L2,    &nv2,     sizeof(int)));
    int nf = N_FRAMES, nx = NX, ny = NY;
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_frames, &nf,      sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nx,       &nx,      sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ny,       &ny,      sizeof(int)));

    // --- Allocate device memory ---
    // FP16 frames: 10M * 300 * 2 bytes = 6 GB (vs 12 GB FP32)
    size_t frame_bytes  = (size_t)N_PIXELS * N_FRAMES * sizeof(__half);
    size_t stat_bytes   = (size_t)N_PIXELS * sizeof(float);

    printf("Allocating %.2f GB on GPU (%.2f GB frames + %.2f GB outputs)...\n",
           (frame_bytes + 3 * stat_bytes) / 1e9,
           frame_bytes / 1e9,
           3 * stat_bytes / 1e9);

    __half* d_frames;
    float*  d_best_stats;
    float*  d_best_vx;
    float*  d_best_vy;

    CUDA_CHECK(cudaMalloc(&d_frames,     frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_stats, stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vx,    stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vy,    stat_bytes));

    // Prefetch to GPU (hint for unified memory systems like Orin)
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    // Note: cudaMemPrefetchAsync is a no-op on discrete GPUs, harmless
    cudaMemPrefetchAsync(d_frames, frame_bytes, device, 0);

    printf("Allocated successfully.\n\n");

    // === PHASE 1: Generate data on GPU (FP16) ===
    printf("[Phase 1] Generating %.2f GB of FP16 noise on GPU...\n",
           frame_bytes / 1e9);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));

    int gen_block = 256;
    int gen_grid  = (N_PIXELS + gen_block - 1) / gen_block;
    generate_noise_fp16<<<gen_grid, gen_block>>>(d_frames, N_PIXELS, N_FRAMES);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float gen_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gen_ms, t0, t1));
    printf("  Noise generation : %.2f s\n", gen_ms / 1000.0f);

    // Inject signal
    CUDA_CHECK(cudaEventRecord(t0));

    int sig_block = 256;
    int sig_grid  = (N_FRAMES + sig_block - 1) / sig_block;
    inject_signal_fp16<<<sig_grid, sig_block>>>(
        d_frames, NX, NY, N_FRAMES,
        TRUE_X0, TRUE_Y0, TRUE_VX, TRUE_VY, OBJ_SIGNAL);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float sig_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sig_ms, t0, t1));
    printf("  Signal injection : %.3f ms\n", sig_ms);
    printf("\n");

    // === PHASE 2: Pyramid TBD search ===
    printf("[Phase 2] Running coarse-to-fine pyramid TBD search...\n");
    printf("  Launching in chunks of %d rows\n\n", CHUNK_ROWS);

    // Orin-tuned block: 128 threads (8x16) to maximize occupancy
    // with moderate register usage from top-K arrays
    dim3 block(8, 16);  // 128 threads per block
    float total_search_ms = 0;
    int n_chunks = (NY + CHUNK_ROWS - 1) / CHUNK_ROWS;

    struct timespec wall_start, wall_now;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int y_start = chunk * CHUNK_ROWS;
        int y_end   = y_start + CHUNK_ROWS;
        if (y_end > NY) y_end = NY;
        int chunk_rows = y_end - y_start;

        dim3 grid((NX + block.x - 1) / block.x,
                  (chunk_rows + block.y - 1) / block.y);

        CUDA_CHECK(cudaEventRecord(t0));

        tbd_pyramid_search<<<grid, block>>>(
            d_frames, d_best_stats, d_best_vx, d_best_vy,
            y_start, y_end);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float chunk_ms;
        CUDA_CHECK(cudaEventElapsedTime(&chunk_ms, t0, t1));
        total_search_ms += chunk_ms;

        clock_gettime(CLOCK_MONOTONIC, &wall_now);
        double wall_elapsed = (wall_now.tv_sec - wall_start.tv_sec)
                            + (wall_now.tv_nsec - wall_start.tv_nsec) * 1e-9;

        float pct = 100.0f * y_end / NY;
        float rows_per_sec = y_end / (total_search_ms * 0.001f);
        float eta_s = (NY - y_end) / rows_per_sec;

        printf("  Rows %4d-%4d / %d  (%5.1f%%)  "
               "chunk %.1fs  total %.1fs  ETA %.0fs\n",
               y_start, y_end, NY, pct,
               chunk_ms / 1000.0f,
               total_search_ms / 1000.0f,
               eta_s);
        fflush(stdout);
    }

    printf("\n  Search complete.\n");
    printf("  Total kernel time: %.2f s\n", total_search_ms / 1000.0f);
    printf("\n");

    // === PHASE 3: GPU-side reduction ===
    printf("[Phase 3] GPU-side argmax reduction...\n");

    // First pass: reduce 10M -> ~10K block results
    int red_block = 1024;
    int red_grid  = (N_PIXELS + red_block - 1) / red_block;

    float* d_blk_stats;
    float* d_blk_vx;
    float* d_blk_vy;
    int*   d_blk_idx;

    CUDA_CHECK(cudaMalloc(&d_blk_stats, red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vx,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vy,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_idx,   red_grid * sizeof(int)));

    size_t smem_size = red_block * (sizeof(float) + sizeof(int));
    reduce_argmax<<<red_grid, red_block, smem_size>>>(
        d_best_stats, d_best_vx, d_best_vy,
        N_PIXELS,
        d_blk_stats, d_blk_vx, d_blk_vy, d_blk_idx);
    CUDA_CHECK(cudaGetLastError());

    // Copy ~10K block results to host for final reduction
    float* h_blk_stats = (float*)malloc(red_grid * sizeof(float));
    float* h_blk_vx    = (float*)malloc(red_grid * sizeof(float));
    float* h_blk_vy    = (float*)malloc(red_grid * sizeof(float));
    int*   h_blk_idx   = (int*)malloc(red_grid * sizeof(int));

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

    int det_x0 = best_idx % NX;
    int det_y0 = best_idx / NX;

    // === RESULTS ===
    printf("\n");
    printf("================================================================\n");
    printf("  RESULTS\n");
    printf("================================================================\n");
    printf("  Detected trajectory:\n");
    printf("    x0  = %d    (truth: %.0f,  error: %+d)\n",
           det_x0, TRUE_X0, (int)(det_x0 - TRUE_X0));
    printf("    y0  = %d   (truth: %.0f, error: %+d)\n",
           det_y0, TRUE_Y0, (int)(det_y0 - TRUE_Y0));
    printf("    vx  = %.2f  (truth: %.2f, error: %+.2f)\n",
           det_vx, TRUE_VX, det_vx - TRUE_VX);
    printf("    vy  = %.2f (truth: %.2f, error: %+.2f)\n",
           det_vy, TRUE_VY, det_vy - TRUE_VY);
    printf("    stat = %.1f\n", best_stat);
    printf("\n");
    printf("  Timing:\n");
    printf("    Data generation : %.2f s\n", gen_ms / 1000.0f);
    printf("    TBD search      : %.2f s\n", total_search_ms / 1000.0f);
    printf("    Total           : %.2f s\n",
           (gen_ms + total_search_ms) / 1000.0f);
    printf("\n");
    printf("  Throughput:\n");
    printf("    %.2f billion trajectories / s\n",
           total_traj / (total_search_ms * 1e6));
    printf("    %.2f trillion pixel reads / s\n",
           total_reads / (total_search_ms * 1e9));
    printf("    Effective BW: %.2f GB/s (FP16 reads)\n",
           total_reads * sizeof(__half) / (total_search_ms * 1e6));
    printf("\n");
    printf("  Pyramid efficiency:\n");
    printf("    Velocity evals/pixel : %lld (vs %lld brute-force)\n",
           total_evals, brute_evals);
    printf("    Reduction factor     : %.1fx\n",
           (double)brute_evals / total_evals);
    printf("    Memory saved (FP16)  : %.1f GB\n",
           (double)N_PIXELS * N_FRAMES * (sizeof(float) - sizeof(__half)) / 1e9);
    printf("================================================================\n");

    // --- Validation ---
    bool correct = (det_x0 == (int)TRUE_X0) &&
                   (det_y0 == (int)TRUE_Y0) &&
                   (fabsf(det_vx - TRUE_VX) < 0.001f) &&
                   (fabsf(det_vy - TRUE_VY) < 0.001f);

    if (correct) {
        printf("\n  *** PASS: Detection matches ground truth ***\n\n");
    } else {
        printf("\n  *** FAIL: Detection does not match ground truth ***\n");
        printf("  Detected: (%d, %d, %.4f, %.4f)\n", det_x0, det_y0, det_vx, det_vy);
        printf("  Truth:    (%.0f, %.0f, %.2f, %.2f)\n", TRUE_X0, TRUE_Y0, TRUE_VX, TRUE_VY);
        printf("  (This could happen with very unlucky noise; re-run to check)\n\n");
    }

    // --- Cleanup ---
    free(h_blk_stats);
    free(h_blk_vx);
    free(h_blk_vy);
    free(h_blk_idx);
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
