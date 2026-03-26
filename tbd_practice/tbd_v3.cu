/**
 * TBD v3 — Coarse-to-Fine Velocity Pyramid
 * =========================================================
 *
 *  CHANGE 1: INT8 frame storage
 *    Frames stored as uint8_t (was FP16). Per-frame linear scale:
 *      scale[t] = max_val_in_frame[t] / 255.0
 *      dequant : fp32 = u8 * scale[t]
 *    Scales held in device global memory via __device__ float* g_frame_scale
 *    (not constant memory, so frame count is unbounded). All warp lanes read
 *    the same scale[t] per loop iteration → L2 broadcast, no penalty vs
 *    constant memory. Halves memory traffic vs FP16.
 *    Assumes non-negative frame values (radar/optical intensity).
 *
 *  CHANGE 2: L0 shift-and-stack (velocity-outer, pixel-inner)
 *    Replaces the old warp-per-pixel L0 search. For each (vx, vy) pair:
 *      shift_and_accumulate  — one thread per pixel computes
 *                              H[y,x] = Σ_t frame[t][y+vy*t][x+vx*t]
 *                              Adjacent threads read adjacent columns
 *                              → fully coalesced 128-byte cache lines
 *                              (~6 registers/thread, ~1% vs ~64× waste before)
 *      topk_update_L0        — one thread per pixel inserts H[pixel]
 *                              into that pixel's TOP_K candidate list
 *    Launched nv0² times (one per coarse velocity pair) with init_topk
 *    initialising d_topk_L0 to -FLT_MAX before the loop.
 *
 *  L1 / L2: unchanged — warp-per-pixel (Strategy 3 + 4)
 *    tbd_warp_L1  reads d_topk_L0, writes d_topk_L1
 *    tbd_warp_L2  reads d_topk_L1, writes best_stats/vx/vy
 *    Per-level register budgets: L1/L2 ~35-40/thread → ~50% occupancy
 *
 * Compile (Orin, sm_87):
 *   nvcc -O3 -arch=sm_87 --use_fast_math -o tbd_v3 tbd_v3.cu
 *
 * Compile (RTX 4090, sm_89):
 *   nvcc -O3 -arch=sm_89 --use_fast_math -o tbd_v3 tbd_v3.cu
 *
 * Run:
 *   ./tbd_v3 tbd_frames.bin tbd_params.bin
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <cstdint>
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

// === Pyramid levels ===
// Level 0 (coarse): step=0.50, covers full range, 15x15 = 225 pairs
#define L0_STEP     0.50f
// Level 1 (medium): step=0.10, +/-0.50 around L0 winners, 11x11 = 121 pairs
#define L1_STEP     0.10f
#define L1_RADIUS   0.50f
// Level 2 (fine):   step=0.05, +/-0.10 around L1 winners, 5x5 = 25 pairs
#define L2_STEP     0.05f
#define L2_RADIUS   0.10f

// How many top candidates to carry between pyramid levels
#define TOP_K       6

// Strategy 4: number of pixels (warps) packed into one thread block.
// blockDim = (32, PIXELS_PER_BLOCK) = 128 threads/block at PIXELS_PER_BLOCK=4.
// Tunable: larger values improve block-level efficiency but may hurt
// occupancy if register file per block grows too large.
#define PIXELS_PER_BLOCK  4

// ============================================================
// Constant memory for velocity grids at each pyramid level
// ============================================================
#define MAX_VEL_L0  32
#define MAX_VEL_L1  32
#define MAX_VEL_L2  32

__constant__ float d_vel_L0[MAX_VEL_L0];
__constant__ float d_vel_L1[MAX_VEL_L1];  // Relative offsets [0, 2*L1_RADIUS]
__constant__ float d_vel_L2[MAX_VEL_L2];  // Relative offsets [0, 2*L2_RADIUS]

// INT8 dequant scales: fp32 = u8 * g_frame_scale[t]
// Stored in device global memory (not constant) so nf is unbounded.
// All warp lanes read the same t simultaneously → L2 broadcast, same cost
// as constant memory for this uniform-access pattern.
__device__ float* g_frame_scale;

__constant__ int   d_nv_L0;
__constant__ int   d_nv_L1;
__constant__ int   d_nv_L2;
__constant__ int   d_n_frames;
__constant__ int   d_nx;
__constant__ int   d_ny;
__constant__ float d_v_min;
__constant__ float d_v_max;

// ============================================================
// Device helper: evaluate one trajectory hypothesis
// Reads INT8 (uint8_t), dequantizes with per-frame scale,
// accumulates in FP32 for precision
// ============================================================
__device__ __forceinline__ float evaluate_trajectory(
    const uint8_t* __restrict__ frames,
    int x0, int y0, float vx, float vy,
    int nx, int ny, int nf)
{
    const int frame_stride = ny * nx;
    float stat = 0.0f;

    for (int t = 0; t < nf; t++) {
        int px = __float2int_rn((float)x0 + vx * (float)t);
        int py = __float2int_rn((float)y0 + vy * (float)t);

        if (px < 0 || px >= nx || py < 0 || py >= ny)
            return -FLT_MAX;

        stat += (float)__ldg(&frames[(size_t)t * frame_stride + py * nx + px])
                * g_frame_scale[t];
    }
    return stat;
}

// ============================================================
// Struct: one candidate trajectory result per pixel per level
// Stored in global memory between kernel launches (Strategy 3)
// ============================================================
struct Candidate {
    float stat;
    float vx;
    float vy;
};

// ============================================================
// Device helper: warp-shuffle top-K extraction (Strategy 4)
// ============================================================
// Called by ALL 32 threads in the warp after their velocity loops.
// Each lane brings its single local best (local_stat/vx/vy).
// Performs TOP_K rounds of "find warp-max, record winner, exclude"
// using __shfl_down_sync + __ballot_sync; no shared memory needed.
// Only lane 0 writes the TOP_K results to out_topk[].
//
// local_stat is passed by reference so winners can be excluded
// from subsequent rounds by setting their value to -FLT_MAX.
// ============================================================
__device__ void warp_topk_write(
    Candidate* __restrict__ out_topk,
    float& local_stat,
    float  local_vx,
    float  local_vy,
    int    lane)
{
    for (int k = 0; k < TOP_K; k++) {
        // Step 1: warp reduce to find global max
        float s = local_stat;
        for (int off = 16; off > 0; off >>= 1)
            s = fmaxf(s, __shfl_down_sync(0xffffffff, s, off));
        float global_max = __shfl_sync(0xffffffff, s, 0);

        // Step 2: find the lowest-numbered lane holding that max
        unsigned ballot = __ballot_sync(0xffffffff, local_stat == global_max);
        int winner = (ballot != 0u) ? (__ffs(ballot) - 1) : 0;

        // Step 3: broadcast winner's vx/vy to all lanes (all must call shfl)
        float w_vx = __shfl_sync(0xffffffff, local_vx, winner);
        float w_vy = __shfl_sync(0xffffffff, local_vy, winner);

        // Step 4: lane 0 records this round's winner
        if (lane == 0) {
            out_topk[k].stat = global_max;
            out_topk[k].vx   = w_vx;
            out_topk[k].vy   = w_vy;
        }

        // Step 5: exclude winner from subsequent rounds
        if (lane == winner) local_stat = -FLT_MAX;
    }
}

// ============================================================
// Kernel: Level-1 warp-per-pixel refinement  (Strategy 3 + 4)
// ============================================================
// Reads d_topk_L0 (TOP_K entries per pixel), searches an
// 11×11 neighbourhood (step L1_STEP, radius L1_RADIUS) around
// each L0 winner. The TOP_K × nv1² search pairs are striped
// across 32 lanes; each lane tracks 1 local best.
//
// Register budget: ~35-40/thread  (one level's state at a time,
//                  plus L0 cache — no L2 state in flight)
// ============================================================
__global__ __launch_bounds__(32 * PIXELS_PER_BLOCK, 4)
void tbd_warp_L1(
    const uint8_t*    __restrict__ frames,
    const Candidate*  __restrict__ d_topk_L0,   // input
    Candidate*        __restrict__ d_topk_L1)   // output
{
    const int lane      = threadIdx.x;
    const int warp_id   = threadIdx.y;
    const int pixel_idx = blockIdx.x * PIXELS_PER_BLOCK + warp_id;

    const int nx = d_nx, ny = d_ny, nf = d_n_frames;
    if (pixel_idx >= nx * ny) return;

    const int x0 = pixel_idx % nx;
    const int y0 = pixel_idx / nx;

    const float inv_T = 1.f / (float)(nf - 1);
    const float vx_lo = fmaxf(d_v_min, -(float)x0 * inv_T);
    const float vx_hi = fminf(d_v_max,  (float)(nx - 1 - x0) * inv_T);
    const float vy_lo = fmaxf(d_v_min, -(float)y0 * inv_T);
    const float vy_hi = fminf(d_v_max,  (float)(ny - 1 - y0) * inv_T);

    // All 32 lanes in the warp read the same pixel's L0 top-K entries.
    // This is a broadcast from L2 cache (same address for all lanes).
    const Candidate* src = d_topk_L0 + pixel_idx * TOP_K;
    Candidate l0[TOP_K];
    for (int k = 0; k < TOP_K; k++) l0[k] = src[k];

    const int nv1         = d_nv_L1;
    const int pairs_per_c = nv1 * nv1;           // pairs per L0 candidate
    const int n_vel_pairs = TOP_K * pairs_per_c;  // total search space

    float local_stat = -FLT_MAX;
    float local_vx   = 0.f;
    float local_vy   = 0.f;

    for (int vel_idx = lane; vel_idx < n_vel_pairs; vel_idx += 32) {
        const int c  = vel_idx / pairs_per_c;
        const int vi = (vel_idx % pairs_per_c) / nv1;
        const int vj = vel_idx % nv1;

        if (l0[c].stat <= -FLT_MAX) continue;

        const float vx = l0[c].vx - L1_RADIUS + d_vel_L1[vi];
        const float vy = l0[c].vy - L1_RADIUS + d_vel_L1[vj];

        if (vx < vx_lo || vx > vx_hi || vy < vy_lo || vy > vy_hi) continue;

        float stat = evaluate_trajectory(frames, x0, y0, vx, vy, nx, ny, nf);
        if (stat > local_stat) {
            local_stat = stat;
            local_vx   = vx;
            local_vy   = vy;
        }
    }

    warp_topk_write(d_topk_L1 + pixel_idx * TOP_K,
                    local_stat, local_vx, local_vy, lane);
}

// ============================================================
// Kernel: Level-2 warp-per-pixel fine refinement  (Strategy 3 + 4)
// ============================================================
// Reads d_topk_L1, searches a 5×5 neighbourhood (step L2_STEP,
// radius L2_RADIUS) around each L1 winner. Because L2 only needs
// the single global best per pixel (not a new top-K list), the
// warp reduction reduces to a simple warp-max with one winner.
//
// Register budget: ~35-40/thread
// ============================================================
__global__ __launch_bounds__(32 * PIXELS_PER_BLOCK, 4)
void tbd_warp_L2(
    const uint8_t*   __restrict__ frames,
    const Candidate* __restrict__ d_topk_L1,    // input
    float*           __restrict__ best_stats,   // [n_pixels] output
    float*           __restrict__ best_vx_out,
    float*           __restrict__ best_vy_out)
{
    const int lane      = threadIdx.x;
    const int warp_id   = threadIdx.y;
    const int pixel_idx = blockIdx.x * PIXELS_PER_BLOCK + warp_id;

    const int nx = d_nx, ny = d_ny, nf = d_n_frames;
    if (pixel_idx >= nx * ny) return;

    const int x0 = pixel_idx % nx;
    const int y0 = pixel_idx / nx;

    const float inv_T = 1.f / (float)(nf - 1);
    const float vx_lo = fmaxf(d_v_min, -(float)x0 * inv_T);
    const float vx_hi = fminf(d_v_max,  (float)(nx - 1 - x0) * inv_T);
    const float vy_lo = fmaxf(d_v_min, -(float)y0 * inv_T);
    const float vy_hi = fminf(d_v_max,  (float)(ny - 1 - y0) * inv_T);

    const Candidate* src = d_topk_L1 + pixel_idx * TOP_K;
    Candidate l1[TOP_K];
    for (int k = 0; k < TOP_K; k++) l1[k] = src[k];

    const int nv2         = d_nv_L2;
    const int pairs_per_c = nv2 * nv2;
    const int n_vel_pairs = TOP_K * pairs_per_c;

    float local_stat = -FLT_MAX;
    float local_vx   = 0.f;
    float local_vy   = 0.f;

    for (int vel_idx = lane; vel_idx < n_vel_pairs; vel_idx += 32) {
        const int c  = vel_idx / pairs_per_c;
        const int vi = (vel_idx % pairs_per_c) / nv2;
        const int vj = vel_idx % nv2;

        if (l1[c].stat <= -FLT_MAX) continue;

        const float vx = l1[c].vx - L2_RADIUS + d_vel_L2[vi];
        const float vy = l1[c].vy - L2_RADIUS + d_vel_L2[vj];

        if (vx < vx_lo || vx > vx_hi || vy < vy_lo || vy > vy_hi) continue;

        float stat = evaluate_trajectory(frames, x0, y0, vx, vy, nx, ny, nf);
        if (stat > local_stat) {
            local_stat = stat;
            local_vx   = vx;
            local_vy   = vy;
        }
    }

    // Warp reduce to find single global best (top-1 variant)
    float s = local_stat;
    for (int off = 16; off > 0; off >>= 1)
        s = fmaxf(s, __shfl_down_sync(0xffffffff, s, off));
    float best_s = __shfl_sync(0xffffffff, s, 0);

    unsigned ballot = __ballot_sync(0xffffffff, local_stat == best_s);
    int winner      = (ballot != 0u) ? (__ffs(ballot) - 1) : 0;
    float w_vx      = __shfl_sync(0xffffffff, local_vx, winner);
    float w_vy      = __shfl_sync(0xffffffff, local_vy, winner);

    if (lane == 0) {
        best_stats[pixel_idx]  = best_s;
        best_vx_out[pixel_idx] = w_vx;
        best_vy_out[pixel_idx] = w_vy;
    }
}

// ============================================================
// Kernel: Shift-and-Stack — L0 coarse search (velocity-outer)
// ============================================================
// For fixed (vx, vy), computes H[y,x] = Σ_t frame[t][y+vy*t][x+vx*t]
// for all pixels simultaneously. Adjacent threads read adjacent columns
// → fully coalesced 128-byte cache lines.
// Frame data is INT8 (uint8_t); dequantized via g_frame_scale[t].
// Register usage: ~6/thread. Launched once per (vx,vy) pair.
// ============================================================
__global__ void shift_and_accumulate(
    const uint8_t* __restrict__ frames,
    float* __restrict__ H,          // [ny × nx] output
    float vx, float vy,
    int nx, int ny, int nf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    float acc = 0.f;
    const int frame_stride = ny * nx;
    for (int t = 0; t < nf; t++) {
        int px = __float2int_rn(x + vx * t);
        int py = __float2int_rn(y + vy * t);
        if (px < 0 || px >= nx || py < 0 || py >= ny) { acc = -FLT_MAX; break; }
        acc += (float)__ldg(&frames[(size_t)t * frame_stride + py * nx + px])
               * g_frame_scale[t];
    }
    H[y * nx + x] = acc;
}

// ============================================================
// Kernel: Initialize top-K candidate array to -FLT_MAX
// ============================================================
__global__ void init_topk(Candidate* __restrict__ topk, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { topk[i].stat = -FLT_MAX; topk[i].vx = 0.f; topk[i].vy = 0.f; }
}

// ============================================================
// Kernel: Update per-pixel top-K from one shift image H
// ============================================================
// Called once per (vx,vy) pair after shift_and_accumulate.
// Each thread owns one pixel; replaces the worst slot in its
// TOP_K array if H[pixel] beats it.
// ============================================================
__global__ void topk_update_L0(
    const float* __restrict__ H,
    float vx, float vy,
    Candidate* __restrict__ d_topk,
    int n_pixels)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pixels) return;

    float s = H[pid];
    if (s <= -FLT_MAX + 1.f) return;   // trajectory left FOV

    Candidate* topk = d_topk + pid * TOP_K;
    int min_k = 0;
    for (int k = 1; k < TOP_K; k++)
        if (topk[k].stat < topk[min_k].stat) min_k = k;
    if (s > topk[min_k].stat) {
        topk[min_k].stat = s;
        topk[min_k].vx   = vx;
        topk[min_k].vy   = vy;
    }
}

// ============================================================
// Kernel: GPU-side argmax reduction (warp-shuffle, no shared mem)
// ============================================================
// Uses warp-shuffle to reduce within each warp, then collects
// warp winners into a 32-element shared buffer for the final
// warp-level pass. Faster and more consistent than shared-mem
// tree reduction (same approach as warp_topk_write in Strategy 4).
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
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    int warp = tid >> 5;

    float s   = (gid < n) ? stats[gid] : -FLT_MAX;
    int   idx = (gid < n) ? gid        : -1;

    // Warp-level reduce: keep max and its original index
    for (int off = 16; off > 0; off >>= 1) {
        float s2   = __shfl_down_sync(0xffffffff, s,   off);
        int   idx2 = __shfl_down_sync(0xffffffff, idx, off);
        if (s2 > s) { s = s2; idx = idx2; }
    }

    // Collect one winner per warp into shared memory
    __shared__ float s_warp_stats[32];
    __shared__ int   s_warp_idx[32];
    if (lane == 0) { s_warp_stats[warp] = s; s_warp_idx[warp] = idx; }
    __syncthreads();

    // Final pass: first warp reduces the warp winners
    if (warp == 0) {
        int nwarps = blockDim.x >> 5;
        s   = (lane < nwarps) ? s_warp_stats[lane] : -FLT_MAX;
        idx = (lane < nwarps) ? s_warp_idx[lane]   : -1;
        for (int off = 16; off > 0; off >>= 1) {
            float s2   = __shfl_down_sync(0xffffffff, s,   off);
            int   idx2 = __shfl_down_sync(0xffffffff, idx, off);
            if (s2 > s) { s = s2; idx = idx2; }
        }
        if (lane == 0) {
            block_stats[blockIdx.x] = s;
            block_idx[blockIdx.x]   = idx;
            block_vx[blockIdx.x]    = (idx >= 0 && idx < n) ? vx_arr[idx] : 0.0f;
            block_vy[blockIdx.x]    = (idx >= 0 && idx < n) ? vy_arr[idx] : 0.0f;
        }
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
    printf("  Orin-Optimized TBD: Strategy 3 (multi-pass) +\n");
    printf("                      Strategy 4 (warp-per-pixel)\n");
    printf("================================================================\n\n");

    // --- GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB memory)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    printf("L2 cache: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("Unified memory: %s\n", prop.unifiedAddressing ? "yes" : "no");
    printf("\n");

    // --- Start total wall-clock timer ---
    auto total_start = std::chrono::steady_clock::now();

    // --- Load data ---
    printf("[Phase 1] Loading frames and params from disk...\n");
    auto load_t0 = std::chrono::steady_clock::now();

    int nx, ny, nf;
    float* h_frames_fp32 = load_frames(frame_file, &nx, &ny, &nf);
    printf("  Loaded frames: %d x %d x %d\n", nx, ny, nf);

    float v_min, v_max;
    int n_vel;
    load_params(param_file, &v_min, &v_max, &n_vel);
    printf("  Velocity grid : v_min=%.2f, v_max=%.2f, n_vel=%d\n",
           v_min, v_max, n_vel);

    auto load_t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
    printf("  Load time: %.2f s\n\n", load_ms / 1000.0);

    int n_pixels = nx * ny;

    // --- Build velocity grids ---
    float h_vel_L0[MAX_VEL_L0], h_vel_L1[MAX_VEL_L1], h_vel_L2[MAX_VEL_L2];

    int nv0 = build_vel_grid(h_vel_L0, MAX_VEL_L0, v_min, v_max, L0_STEP);
    int nv1 = build_vel_grid(h_vel_L1, MAX_VEL_L1, 0.0f, 2.0f * L1_RADIUS, L1_STEP);
    int nv2 = build_vel_grid(h_vel_L2, MAX_VEL_L2, 0.0f, 2.0f * L2_RADIUS, L2_STEP);

    long long evals_L0 = (long long)nv0 * nv0;
    long long evals_L1 = (long long)TOP_K * nv1 * nv1;
    long long evals_L2 = (long long)TOP_K * nv2 * nv2;
    long long total_evals = evals_L0 + evals_L1 + evals_L2;

    int nv_brute = 0;
    for (float v = v_min; v <= v_max + 0.02f; v += 0.05f) nv_brute++;
    long long brute_evals = (long long)nv_brute * nv_brute;

    printf("Pyramid velocity search (Strategy 3+4: one warp per pixel per level):\n");
    printf("  Level 0 (coarse, step=%.2f): %d x %d = %lld pairs  "
           "[%lld lanes × ceil(%lld/32) iters]\n",
           L0_STEP, nv0, nv0, evals_L0, (long long)n_pixels, evals_L0);
    printf("  Level 1 (medium, step=%.2f): %d * %d x %d = %lld pairs\n",
           L1_STEP, TOP_K, nv1, nv1, evals_L1);
    printf("  Level 2 (fine,   step=%.2f): %d * %d x %d = %lld pairs\n",
           L2_STEP, TOP_K, nv2, nv2, evals_L2);
    printf("  Total per pixel   : %lld\n", total_evals);
    printf("  Brute-force ref   : %lld\n", brute_evals);
    printf("  Speedup factor    : %.1fx\n", (double)brute_evals / total_evals);
    printf("\n");

    long long total_traj  = (long long)n_pixels * total_evals;
    long long total_reads = total_traj * nf;

    printf("Problem size:\n");
    printf("  Image          : %d x %d = %d pixels (%.1f MP)\n",
           nx, ny, n_pixels, n_pixels / 1e6);
    printf("  Frames         : %d\n", nf);
    printf("  Total traj     : %.2f billion (vs %.1f B brute-force)\n",
           total_traj / 1e9, (double)n_pixels * brute_evals / 1e9);
    printf("  Frame data     : %.2f GB (INT8)\n",
           (double)n_pixels * nf * sizeof(uint8_t) / 1e9);
    printf("  TopK buffers   : %.2f MB (d_topk_L0 + d_topk_L1)\n",
           2.0 * n_pixels * TOP_K * sizeof(Candidate) / 1e6);
    printf("\n");

    // --- Upload constants ---
    CUDA_CHECK(cudaMemcpyToSymbol(d_vel_L0,   h_vel_L0, nv0 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_vel_L1,   h_vel_L1, nv1 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_vel_L2,   h_vel_L2, nv2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nv_L0,    &nv0,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nv_L1,    &nv1,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nv_L2,    &nv2,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_frames, &nf,      sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nx,       &nx,      sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ny,       &ny,      sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_v_min,    &v_min,   sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_v_max,    &v_max,   sizeof(float)));

    // --- Allocate device memory ---
    size_t frame_bytes = (size_t)n_pixels * nf * sizeof(uint8_t);
    size_t stat_bytes  = (size_t)n_pixels * sizeof(float);
    size_t topk_bytes  = (size_t)n_pixels * TOP_K * sizeof(Candidate);

    printf("Allocating %.2f GB on GPU...\n",
           (frame_bytes + 4 * stat_bytes + 2 * topk_bytes) / 1e9);

    uint8_t*   d_frames;
    float*     d_best_stats;
    float*     d_best_vx;
    float*     d_best_vy;
    float*     d_H;         // shift-and-stack intermediate image
    float*     d_scale_arr; // per-frame INT8 dequant scales (device global memory)
    Candidate* d_topk_L0;
    Candidate* d_topk_L1;

    CUDA_CHECK(cudaMalloc(&d_frames,     frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_stats, stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vx,    stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vy,    stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_H,          stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_scale_arr,  nf * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_L0,    topk_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_L1,    topk_bytes));
    printf("Allocated successfully.\n\n");

    // --- Convert FP32 → INT8 (per-frame linear scale) and upload ---
    printf("Converting FP32 -> INT8 (per-frame scale) and uploading to GPU...\n");
    auto up_t0 = std::chrono::steady_clock::now();

    size_t n_total = (size_t)n_pixels * nf;
    float*   h_frame_scales = (float*)malloc(nf * sizeof(float));
    uint8_t* h_frames_u8    = (uint8_t*)malloc(n_total);

    // Per-frame: find max, compute scale, quantize to [0, 255].
    // Assumes non-negative frame values (radar/optical intensity).
    for (int t = 0; t < nf; t++) {
        const float* src = h_frames_fp32 + (size_t)t * n_pixels;
        float max_val = 1e-30f;
        for (int i = 0; i < n_pixels; i++)
            max_val = fmaxf(max_val, src[i]);
        float scale = max_val / 255.0f;
        h_frame_scales[t] = scale;
        uint8_t* dst = h_frames_u8 + (size_t)t * n_pixels;
        for (int i = 0; i < n_pixels; i++)
            dst[i] = (uint8_t)fminf(255.f, fmaxf(0.f, roundf(src[i] / scale)));
    }
    free(h_frames_fp32);
    h_frames_fp32 = nullptr;

    CUDA_CHECK(cudaMemcpy(d_frames,    h_frames_u8,    frame_bytes,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_arr, h_frame_scales, nf * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(g_frame_scale, &d_scale_arr, sizeof(float*)));
    free(h_frames_u8);
    free(h_frame_scales);

    auto up_t1 = std::chrono::steady_clock::now();
    double up_ms = std::chrono::duration<double, std::milli>(up_t1 - up_t0).count();
    printf("  Done (%.2f s)\n\n", up_ms / 1000.0);

    // ================================================================
    // PHASE 2: Three-level warp-per-pixel pyramid search
    // ================================================================
    // Block layout: (32, PIXELS_PER_BLOCK) = 128 threads per block.
    // threadIdx.x = lane within the warp (0..31)
    // threadIdx.y = which pixel this warp handles within the block
    // Each block covers PIXELS_PER_BLOCK consecutive pixels.
    // ================================================================
    const dim3 block(32, PIXELS_PER_BLOCK);
    const int  n_warp_blocks = (n_pixels + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK;

    printf("[Phase 2] Pyramid search — L0 shift-and-stack, L1/L2 warp-per-pixel\n");
    printf("  L1/L2 block: (32 lanes × %d pixels) = %d threads\n",
           PIXELS_PER_BLOCK, 32 * PIXELS_PER_BLOCK);
    printf("  L1/L2 grid : %d blocks (%d pixels total)\n\n", n_warp_blocks, n_pixels);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // -- L0: shift-and-stack coarse search --
    printf("  [L0] Coarse search (shift-and-stack, %d x %d = %lld pairs)...\n",
           nv0, nv0, evals_L0);
    CUDA_CHECK(cudaEventRecord(t0));

    {
        int init_n = n_pixels * TOP_K;
        int init_block = 256;
        int init_grid  = (init_n + init_block - 1) / init_block;
        init_topk<<<init_grid, init_block>>>(d_topk_L0, init_n);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 sas_block(32, 8);
        dim3 sas_grid((nx + 31) / 32, (ny + 7) / 8);
        int  upd_block = 256;
        int  upd_grid  = (n_pixels + upd_block - 1) / upd_block;

        for (int vi = 0; vi < nv0; vi++) {
            for (int vj = 0; vj < nv0; vj++) {
                shift_and_accumulate<<<sas_grid, sas_block>>>(
                    d_frames, d_H, h_vel_L0[vi], h_vel_L0[vj], nx, ny, nf);
                topk_update_L0<<<upd_grid, upd_block>>>(
                    d_H, h_vel_L0[vi], h_vel_L0[vj], d_topk_L0, n_pixels);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float l0_ms;
    CUDA_CHECK(cudaEventElapsedTime(&l0_ms, t0, t1));
    printf("        done: %.2f ms\n\n", l0_ms);

    // -- L1: medium warp refinement --
    printf("  [L1] Medium refine  (%d × %d x %d = %lld pairs/pixel, "
           "ceil(%lld/32) iters/lane)...\n",
           TOP_K, nv1, nv1, evals_L1, evals_L1);
    CUDA_CHECK(cudaEventRecord(t0));
    tbd_warp_L1<<<n_warp_blocks, block>>>(d_frames, d_topk_L0, d_topk_L1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float l1_ms;
    CUDA_CHECK(cudaEventElapsedTime(&l1_ms, t0, t1));
    printf("        done: %.2f ms\n\n", l1_ms);

    // -- L2: fine warp refinement --
    printf("  [L2] Fine refine    (%d × %d x %d = %lld pairs/pixel, "
           "ceil(%lld/32) iters/lane)...\n",
           TOP_K, nv2, nv2, evals_L2, evals_L2);
    CUDA_CHECK(cudaEventRecord(t0));
    tbd_warp_L2<<<n_warp_blocks, block>>>(
        d_frames, d_topk_L1, d_best_stats, d_best_vx, d_best_vy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float l2_ms;
    CUDA_CHECK(cudaEventElapsedTime(&l2_ms, t0, t1));
    printf("        done: %.2f ms\n\n", l2_ms);

    float total_search_ms = l0_ms + l1_ms + l2_ms;
    printf("  Search complete. Total kernel time: %.2f ms\n\n", total_search_ms);

    // === PHASE 3: GPU-side reduction ===
    printf("[Phase 3] GPU-side argmax reduction...\n");

    int red_block = 1024;
    int red_grid  = (n_pixels + red_block - 1) / red_block;

    float* d_blk_stats;
    float* d_blk_vx;
    float* d_blk_vy;
    int*   d_blk_idx;

    CUDA_CHECK(cudaMalloc(&d_blk_stats, red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vx,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vy,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_idx,   red_grid * sizeof(int)));

    reduce_argmax<<<red_grid, red_block>>>(
        d_best_stats, d_best_vx, d_best_vy,
        n_pixels, d_blk_stats, d_blk_vx, d_blk_vy, d_blk_idx);
    CUDA_CHECK(cudaGetLastError());

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

    int det_x0 = best_idx % nx;
    int det_y0 = best_idx / nx;

    // === RESULTS ===
    printf("\n");
    printf("================================================================\n");
    printf("  RESULTS\n");
    printf("================================================================\n");
    printf("  Detected trajectory:\n");
    printf("    x0   = %d\n", det_x0);
    printf("    y0   = %d\n", det_y0);
    printf("    vx   = %.4f px/frame\n", det_vx);
    printf("    vy   = %.4f px/frame\n", det_vy);
    printf("    stat = %.1f\n", best_stat);
    printf("\n");

    auto total_stop = std::chrono::steady_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(total_stop - total_start).count();

    printf("  Timing:\n");
    printf("    Data loading    : %.2f s\n", load_ms / 1000.0);
    printf("    INT8 conversion : %.2f s\n", up_ms / 1000.0);
    printf("    L0 warp kernel  : %.2f ms  [~20-25 regs/thread]\n", l0_ms);
    printf("    L1 warp kernel  : %.2f ms  [~35-40 regs/thread]\n", l1_ms);
    printf("    L2 warp kernel  : %.2f ms  [~35-40 regs/thread]\n", l2_ms);
    printf("    TBD total       : %.2f ms\n", total_search_ms);
    printf("    Total wall time : %.3f s\n", total_wall_ms / 1000.0);
    printf("\n");
    printf("  Throughput:\n");
    printf("    %.2f billion trajectories / s\n",
           total_traj / (total_search_ms * 1e6));
    printf("    %.2f trillion pixel reads / s\n",
           total_reads / (total_search_ms * 1e9));
    printf("    Effective BW: %.2f GB/s (INT8 reads)\n",
           total_reads * sizeof(uint8_t) / (total_search_ms * 1e6));
    printf("\n");
    printf("  Pyramid efficiency:\n");
    printf("    Velocity evals/pixel : %lld (vs %lld brute-force)\n",
           total_evals, brute_evals);
    printf("    Reduction factor     : %.1fx\n",
           (double)brute_evals / total_evals);
    printf("    Memory saved (INT8)  : %.1f GB\n",
           (double)n_pixels * nf * (sizeof(float) - sizeof(uint8_t)) / 1e9);
    printf("================================================================\n");

    // --- Cleanup ---
    free(h_blk_stats); free(h_blk_vx); free(h_blk_vy); free(h_blk_idx);
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_best_stats));
    CUDA_CHECK(cudaFree(d_best_vx));
    CUDA_CHECK(cudaFree(d_best_vy));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_scale_arr));
    CUDA_CHECK(cudaFree(d_topk_L0));
    CUDA_CHECK(cudaFree(d_topk_L1));
    CUDA_CHECK(cudaFree(d_blk_stats));
    CUDA_CHECK(cudaFree(d_blk_vx));
    CUDA_CHECK(cudaFree(d_blk_vy));
    CUDA_CHECK(cudaFree(d_blk_idx));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    return 0;
}
