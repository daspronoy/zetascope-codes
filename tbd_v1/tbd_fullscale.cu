f/**
 * Full-Scale Brute-Force TBD Benchmark
 * =====================================
 * 10 megapixel images, ~20K velocity hypotheses, 300 frames
 *
 * Self-contained: generates data directly on GPU (no disk I/O for 12 GB).
 * Runs the search in row-chunks with progress reporting to avoid TDR
 * timeout and give wall-clock visibility.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 --use_fast_math -o tbd_fullscale tbd_fullscale.cu
 *   (adjust sm_89 for your GPU: sm_86=RTX3090, sm_89=RTX4090, sm_80=A100)
 *
 * Run:
 *   ./tbd_fullscale
 *
 * Expected runtime: 1-4 minutes on RTX 4090
 * Expected result:  detects ground truth trajectory (2000, 1250, 1.30, -0.70)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <cuda_runtime.h>

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
// Problem parameters
// ============================================================
#define NX          4000        // Image width  (pixels)
#define NY          2500        // Image height (pixels)
#define N_PIXELS    (NX * NY)   // 10,000,000
#define N_FRAMES    300         // Temporal frames
#define BG_MEAN     100.0f      // Background rate (photons/pixel/frame)
#define OBJ_SIGNAL  20.0f       // Object signal per frame

// Ground truth trajectory
#define TRUE_X0     2000.0f
#define TRUE_Y0     1250.0f
#define TRUE_VX     1.30f       // Must be on velocity grid
#define TRUE_VY    -0.70f       // Must be on velocity grid

// Velocity search grid
// Range: [-3.50, +3.50], step 0.05 → 141 per axis → 141*141 = 19,881 pairs
#define V_MIN      -3.50f
#define V_MAX       3.50f
#define V_STEP      0.05f
#define MAX_VEL     256         // Max entries per axis in constant memory

// Search chunking (rows per kernel launch, avoids TDR timeout)
#define CHUNK_ROWS  100

// ============================================================
// Constant memory
// ============================================================
__constant__ float d_vel[MAX_VEL];
__constant__ int   d_n_vel;
__constant__ int   d_n_frames;
__constant__ int   d_nx;
__constant__ int   d_ny;

// ============================================================
// Device-side PRNG (xorshift32 + Box-Muller)
// No curand dependency needed.
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
    return (float)xorshift32(state) * 2.3283064365e-10f; // / 2^32
}

// ============================================================
// Kernel 1: Generate background noise directly on GPU
// ============================================================
// Each thread handles one pixel across all frames.
// Uses Normal(100, 10) ≈ Poisson(100) approximation.
// ============================================================
__global__ void generate_noise_kernel(float* frames, int n_pixels, int n_frames) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= n_pixels) return;

    // Seed from pixel index (different seed per pixel for independence)
    unsigned int state = pixel * 1103515245u + 12345u + 7919u;
    // Warm up the PRNG a few rounds
    for (int i = 0; i < 4; i++) xorshift32(&state);

    for (int t = 0; t < n_frames; t++) {
        // Box-Muller transform: two uniforms → one normal
        float u1 = rand_uniform(&state) + 1e-10f;  // avoid log(0)
        float u2 = rand_uniform(&state);
        float z  = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        float val = BG_MEAN + sqrtf(BG_MEAN) * z;  // Normal(100, 10)
        frames[(size_t)t * n_pixels + pixel] = fmaxf(val, 0.0f);
    }
}

// ============================================================
// Kernel 2: Inject deterministic signal along true trajectory
// ============================================================
__global__ void inject_signal_kernel(float* frames, int nx, int ny, int n_frames,
                                     float x0, float y0, float vx, float vy,
                                     float signal) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_frames) return;

    int px = __float2int_rn(x0 + vx * (float)t);
    int py = __float2int_rn(y0 + vy * (float)t);
    if (px >= 0 && px < nx && py >= 0 && py < ny) {
        frames[(size_t)t * nx * ny + py * nx + px] += signal;
    }
}

// ============================================================
// Kernel 3: Brute-force TBD search
// ============================================================
// One thread per (x0, y0) pixel. Loops over all velocity pairs.
// Launched in row-chunks to avoid timeout and enable progress.
// ============================================================
__global__ void tbd_search_kernel(
    const float* __restrict__ frames,
    float*  best_stats,
    int*    best_vxi,
    int*    best_vyi,
    int     y_start,
    int     y_end)
{
    const int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = y_start + blockIdx.y * blockDim.y + threadIdx.y;

    if (x0 >= d_nx || y0 >= y_end) return;

    const int nx = d_nx;
    const int ny = d_ny;
    const int nf = d_n_frames;
    const int nv = d_n_vel;
    const int frame_stride = ny * nx;

    float local_best = -FLT_MAX;
    int   local_vxi  = 0;
    int   local_vyi  = 0;

    for (int vi = 0; vi < nv; vi++) {
        const float vx = d_vel[vi];

        // Early skip: check if trajectory leaves image at t=0 and t=nf-1
        // If x0 + vx*(nf-1) is way out of bounds AND x0 is also problematic,
        // we can skip. But the inner loop's early-break handles this, so
        // we keep it simple.

        for (int vj = 0; vj < nv; vj++) {
            const float vy = d_vel[vj];

            float stat = 0.0f;
            bool valid = true;

            for (int t = 0; t < nf; t++) {
                int px = __float2int_rn((float)x0 + vx * (float)t);
                int py = __float2int_rn((float)y0 + vy * (float)t);

                if (px < 0 || px >= nx || py < 0 || py >= ny) {
                    valid = false;
                    break;
                }

                stat += __ldg(&frames[(size_t)t * frame_stride + py * nx + px]);
            }

            if (valid && stat > local_best) {
                local_best = stat;
                local_vxi  = vi;
                local_vyi  = vj;
            }
        }
    }

    const int idx = y0 * nx + x0;
    best_stats[idx] = local_best;
    best_vxi[idx]   = local_vxi;
    best_vyi[idx]   = local_vyi;
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("================================================================\n");
    printf("  Full-Scale Brute-Force TBD Benchmark\n");
    printf("================================================================\n\n");

    // --- GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);s
    printf("L2 cache: %.1f MB\n\n", prop.l2CacheSize / 1e6);

    // --- Build velocity grid ---
    int n_vel = 0;
    float h_vel[MAX_VEL];
    for (float v = V_MIN; v <= V_MAX + V_STEP * 0.4f; v += V_STEP) {
        if (n_vel >= MAX_VEL) {
            fprintf(stderr, "Velocity grid exceeds MAX_VEL (%d)\n", MAX_VEL);
            exit(1);
        }
        h_vel[n_vel++] = v;
    }

    long long total_vel   = (long long)n_vel * n_vel;
    long long total_traj  = (long long)N_PIXELS * total_vel;
    long long total_reads = total_traj * N_FRAMES;

    printf("Problem size:\n");
    printf("  Image          : %d x %d = %d pixels (%.1f MP)\n",
           NX, NY, N_PIXELS, N_PIXELS / 1e6);
    printf("  Frames         : %d\n", N_FRAMES);
    printf("  Velocity grid  : %d x %d = %lld hypotheses\n",
           n_vel, n_vel, total_vel);
    printf("  Total traj     : %lld (%.1f billion)\n",
           total_traj, total_traj / 1e9);
    printf("  Total reads    : %lld (%.1f trillion)\n",
           total_reads, total_reads / 1e12);
    printf("  Frame data     : %.2f GB\n",
           (double)N_PIXELS * N_FRAMES * sizeof(float) / 1e9);
    printf("\n");

    printf("Ground truth:\n");
    printf("  Position : (%.0f, %.0f)\n", TRUE_X0, TRUE_Y0);
    printf("  Velocity : (%.2f, %.2f) pix/frame\n", TRUE_VX, TRUE_VY);
    printf("  Per-frame SNR : %.1f\n", OBJ_SIGNAL / sqrtf(BG_MEAN));
    printf("  Stacked SNR   : %.1f\n",
           OBJ_SIGNAL * sqrtf((float)N_FRAMES) / sqrtf(BG_MEAN));
    printf("\n");

    // --- Copy constants ---
    CUDA_CHECK(cudaMemcpyToSymbol(d_vel,      h_vel,  n_vel * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_vel,    &n_vel, sizeof(int)));
    int nf = N_FRAMES, nx = NX, ny = NY;
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_frames, &nf,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nx,       &nx,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ny,       &ny,    sizeof(int)));

    // --- Allocate device memory ---
    size_t frame_bytes = (size_t)N_PIXELS * N_FRAMES * sizeof(float);
    size_t pixel_bytes = (size_t)N_PIXELS * sizeof(float);
    size_t idx_bytes   = (size_t)N_PIXELS * sizeof(int);

    printf("Allocating %.2f GB on GPU...\n", 
           (frame_bytes + pixel_bytes + 2 * idx_bytes) / 1e9);

    float* d_frames;
    float* d_best_stats;
    int*   d_best_vxi;
    int*   d_best_vyi;

    CUDA_CHECK(cudaMalloc(&d_frames,     frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_stats, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vxi,   idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vyi,   idx_bytes));

    printf("Allocated successfully.\n\n");

    // === PHASE 1: Generate data on GPU ===
    printf("[Phase 1] Generating %.2f GB of noise on GPU...\n",
           frame_bytes / 1e9);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));

    int gen_block = 256;
    int gen_grid  = (N_PIXELS + gen_block - 1) / gen_block;
    generate_noise_kernel<<<gen_grid, gen_block>>>(
        d_frames, N_PIXELS, N_FRAMES);
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
    inject_signal_kernel<<<sig_grid, sig_block>>>(
        d_frames, NX, NY, N_FRAMES,
        TRUE_X0, TRUE_Y0, TRUE_VX, TRUE_VY, OBJ_SIGNAL);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float sig_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sig_ms, t0, t1));
    printf("  Signal injection : %.3f ms\n", sig_ms);
    printf("\n");

    // === PHASE 2: Run TBD search in row-chunks ===
    printf("[Phase 2] Running brute-force TBD search...\n");
    printf("  Launching in chunks of %d rows\n\n", CHUNK_ROWS);

    dim3 block(16, 16);
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

        tbd_search_kernel<<<grid, block>>>(
            d_frames, d_best_stats, d_best_vxi, d_best_vyi,
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

    // === PHASE 3: Reduction (host-side, 10M entries is trivial) ===
    printf("[Phase 3] Finding global maximum...\n");

    float* h_stats = (float*)malloc(pixel_bytes);
    int*   h_vxi   = (int*)malloc(idx_bytes);
    int*   h_vyi   = (int*)malloc(idx_bytes);

    CUDA_CHECK(cudaMemcpy(h_stats, d_best_stats, pixel_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vxi,   d_best_vxi,   idx_bytes,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vyi,   d_best_vyi,   idx_bytes,   cudaMemcpyDeviceToHost));

    float best_stat = -FLT_MAX;
    int   best_idx  = -1;
    int   best_vi   = 0;
    int   best_vj   = 0;

    for (int i = 0; i < N_PIXELS; i++) {
        if (h_stats[i] > best_stat) {
            best_stat = h_stats[i];
            best_idx  = i;
            best_vi   = h_vxi[i];
            best_vj   = h_vyi[i];
        }
    }

    int det_x0 = best_idx % NX;
    int det_y0 = best_idx / NX;
    float det_vx = h_vel[best_vi];
    float det_vy = h_vel[best_vj];

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
    printf("    %.2f TB/s effective read bandwidth\n",
           total_reads * sizeof(float) / (total_search_ms * 1e9));
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
        printf("  (This could happen with very unlucky noise; re-run to check)\n\n");
    }

    // --- Cleanup ---
    free(h_stats);
    free(h_vxi);
    free(h_vyi);
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_best_stats));
    CUDA_CHECK(cudaFree(d_best_vxi));
    CUDA_CHECK(cudaFree(d_best_vyi));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    return 0;
}
