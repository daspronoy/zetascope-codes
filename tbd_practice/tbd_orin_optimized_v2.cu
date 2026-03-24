/**
 * Orin-Optimized TBD: All 4 Register-Pressure Mitigation Strategies
 * ==================================================================
 *
 * Targets: NVIDIA Jetson AGX Orin (sm_87) / RTX 5070 Ti (sm_120, CUDA 12.8+)
 *
 * Strategy 1 — SHIFT-AND-STACK (Velocity-Outer Loop)
 *   CPU outer loop over nv0² = 225 L0 velocity pairs.
 *   shift_and_stack: ~6 regs/thread → ~75%+ SM occupancy.
 *   topk_update_L0:  fast-reject path skips most pixels per iteration.
 *
 * Strategy 2 — SPATIAL DOWNSELECTION (CPU + GPU Hybrid)
 *   After L0, CPU uses std::nth_element on per-pixel best L0 scores
 *   to select top-N_ACTIVE_MAX pixels. L1+L2 run only on that subset.
 *   Work reduction for L1+L2: ~n_pixels / N_ACTIVE_MAX (e.g., 200× at 10 MP).
 *
 * Strategy 3 — MULTI-PASS KERNEL SPLIT
 *   L1 is a separate kernel: reads d_topk_L0 from global, writes d_topk_L1.
 *   L2 is a separate kernel: reads d_topk_L1 lazily (one candidate at a time),
 *   tracks only a 3-scalar global best — minimal register pressure.
 *
 * Strategy 4 — WARP-PER-PIXEL FOR L1
 *   32 threads cooperate on one pixel in the L1 kernel.
 *   Each lane tracks 1 local best (3 regs) across its velocity partition.
 *   Warp-shuffle top-K merge (10 rounds × 5 xor-shuffles) gathers top-10
 *   from 32 lane candidates without any shared memory.
 *   Eliminates topk_L1[10] register array: ~3 regs vs ~30 in original.
 *
 * Kernel register budgets (all strategies active):
 *   shift_and_stack : ~6  regs → 16+ blocks/SM   ← dominates L0 compute
 *   topk_update_L0  : ~35 regs → short update pass
 *   tbd_L1_warp     : ~12 regs → high occupancy on dominant L1 work
 *   tbd_L2_pixel    : ~20 regs → small L2 work, near-peak occupancy
 *
 * Compile (Orin, sm_87):
 *   nvcc -O3 -arch=sm_87 --use_fast_math -lineinfo -o tbd tbd_orin_optimized_v2.cu
 *
 * Compile (RTX 5070 Ti, requires CUDA 12.8+):
 *   nvcc -O3 -arch=sm_120 --use_fast_math -lineinfo -o tbd tbd_orin_optimized_v2.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <numeric>       // std::iota
#include <vector>
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

// === Pyramid levels ===
#define L0_STEP    0.50f
#define L1_STEP    0.10f
#define L1_RADIUS  0.50f
#define L2_STEP    0.05f
#define L2_RADIUS  0.10f

#define TOP_K      10

// Strategy 2: maximum number of pixels to refine in L1/L2.
// For Orin-scale (10MP) this gives ~200x work reduction.
// For small test images (256×256 = 65K pixels) it's automatically capped.
#ifndef N_ACTIVE_MAX
#define N_ACTIVE_MAX 50000
#endif

#define MAX_VEL_L0  32
#define MAX_VEL_L1  32
#define MAX_VEL_L2  32

#define WARP_SIZE   32
#define FULL_MASK   0xFFFFFFFFu

// ============================================================
// Constant memory
// ============================================================
__constant__ float d_vel_L0[MAX_VEL_L0];
__constant__ float d_vel_L1[MAX_VEL_L1];
__constant__ float d_vel_L2[MAX_VEL_L2];
__constant__ int   d_nv_L0;
__constant__ int   d_nv_L1;
__constant__ int   d_nv_L2;
__constant__ int   d_n_frames;
__constant__ int   d_nx;
__constant__ int   d_ny;
__constant__ float d_v_min;
__constant__ float d_v_max;

// ============================================================
// Device helper: evaluate one trajectory
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
            return -FLT_MAX;
        stat += __half2float(frames[(size_t)t * frame_stride + py * nx + px]);
    }
    return stat;
}

// ============================================================
// Kernel 1 (Strategy 1): Shift-and-Stack
// ============================================================
// For a fixed (vx, vy), all threads compute in parallel:
//   H[y, x] = Σ_t  frame[t][y + vy*t][x + vx*t]
//
// Register budget: ~6 (acc, stride, px, py, t, bounds check)
// Access pattern: all warp threads read the same frame-t but
// consecutive x → coalesced loads with strong L2 reuse.
// ============================================================
__global__ __launch_bounds__(256)
void shift_and_stack(
    const __half* __restrict__ frames,
    float* __restrict__ H,
    float vx, float vy)
{
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int nx = d_nx, ny = d_ny, nf = d_n_frames;
    if (x >= nx || y >= ny) return;

    const int frame_stride = ny * nx;
    float acc = 0.0f;
    for (int t = 0; t < nf; t++) {
        int px = __float2int_rn((float)x + vx * (float)t);
        int py = __float2int_rn((float)y + vy * (float)t);
        if (px < 0 || px >= nx || py < 0 || py >= ny) { acc = -FLT_MAX; break; }
        acc += __half2float(frames[(size_t)t * frame_stride + py * nx + px]);
    }
    H[y * nx + x] = acc;
}

// ============================================================
// Kernel 2 (Strategy 1): Per-pixel top-K update after each shift image
// ============================================================
// Layout: d_topk_stat[slot * n_pixels + pid]  — slot-major
//   → consecutive threads access consecutive pixels per slot → coalesced.
//
// Fast-reject: read only the worst slot (slot K-1) first.
// The majority of pixels will be rejected without loading all K entries.
// ============================================================
__global__ __launch_bounds__(1024)
void topk_update_L0(
    const float* __restrict__ H,
    float* __restrict__ d_topk_stat,   // [TOP_K × n_pixels]
    float* __restrict__ d_topk_vx,
    float* __restrict__ d_topk_vy,
    int n_pixels,
    float vx, float vy)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pixels) return;

    const float h = H[pid];
    if (h <= d_topk_stat[(TOP_K - 1) * n_pixels + pid]) return;  // fast reject

    // Read all K slots into registers
    float s[TOP_K], tvx[TOP_K], tvy[TOP_K];
    for (int k = 0; k < TOP_K; k++) {
        s[k]   = d_topk_stat[k * n_pixels + pid];
        tvx[k] = d_topk_vx  [k * n_pixels + pid];
        tvy[k] = d_topk_vy  [k * n_pixels + pid];
    }

    // Insertion sort (descending)
    int pos = TOP_K - 1;
    while (pos > 0 && h > s[pos - 1]) {
        s[pos]   = s[pos - 1];
        tvx[pos] = tvx[pos - 1];
        tvy[pos] = tvy[pos - 1];
        pos--;
    }
    s[pos] = h;  tvx[pos] = vx;  tvy[pos] = vy;

    // Write back only the shifted suffix
    for (int k = pos; k < TOP_K; k++) {
        d_topk_stat[k * n_pixels + pid] = s[k];
        d_topk_vx  [k * n_pixels + pid] = tvx[k];
        d_topk_vy  [k * n_pixels + pid] = tvy[k];
    }
}

// ============================================================
// Kernel 3 (Strategies 3 + 4): L1 refinement — warp-per-pixel
// ============================================================
// Each warp of 32 threads cooperates on ONE active pixel.
// The nv1² L1 velocity pairs per L0 candidate are distributed
// round-robin across the 32 lanes: lane i handles pairs i, i+32, ...
// Each lane tracks its single local best (3 registers).
//
// After the search loops, a warp-shuffle top-K merge selects the
// global top-10 from 32 lane candidates — no shared memory needed.
// Each round: 5 xor-shuffles (max reduce) + 1 ballot + 1 shfl_sync
// × TOP_K = 10 rounds total.
//
// d_topk_L1 layout: [slot × n_active + active_slot]  — slot-major,
// indexed by active_slot within the compact pixel array.
// Lane r writes the r-th top-K slot directly to global memory.
// ============================================================
__global__ __launch_bounds__(128)
void tbd_L1_warp(
    const __half* __restrict__ frames,
    const float* __restrict__ d_topk_L0_stat,   // [TOP_K × n_pixels]
    const float* __restrict__ d_topk_L0_vx,
    const float* __restrict__ d_topk_L0_vy,
    float* __restrict__ d_topk_L1_stat,          // [TOP_K × n_active]
    float* __restrict__ d_topk_L1_vx,
    float* __restrict__ d_topk_L1_vy,
    const int* __restrict__ d_active_pixels,
    int n_active,
    int n_pixels)                                 // full image size for L0 indexing
{
    // 1 warp = 1 pixel; 128 threads/block = 4 warps = 4 pixels/block
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int n_warps = blockDim.x  / WARP_SIZE;
    const int slot    = blockIdx.x * n_warps + warp_id;  // active pixel slot

    if (slot >= n_active) return;

    const int pid = d_active_pixels[slot];
    const int nx  = d_nx, ny = d_ny, nf = d_n_frames;
    const int nv1 = d_nv_L1;
    const int x0  = pid % nx;
    const int y0  = pid / nx;

    // Per-pixel valid velocity range
    const float inv_T = 1.0f / (float)(nf - 1);
    const float vx_lo = fmaxf(d_v_min, -(float)x0 * inv_T);
    const float vx_hi = fminf(d_v_max,  (float)(nx - 1 - x0) * inv_T);
    const float vy_lo = fmaxf(d_v_min, -(float)y0 * inv_T);
    const float vy_hi = fminf(d_v_max,  (float)(ny - 1 - y0) * inv_T);

    // ---- Search phase: each lane tracks 1 local best ----
    float local_stat = -FLT_MAX;
    float local_vx   = 0.f;
    float local_vy   = 0.f;

    const int nv1sq = nv1 * nv1;

    for (int c = 0; c < TOP_K; c++) {
        // All 32 lanes read the same address → L2 broadcast (1 cache line)
        const float l0s = d_topk_L0_stat[c * n_pixels + pid];
        if (l0s <= -FLT_MAX) continue;  // uniform branch: no warp divergence

        const float cvx = d_topk_L0_vx[c * n_pixels + pid];
        const float cvy = d_topk_L0_vy[c * n_pixels + pid];

        // Distribute nv1² pairs round-robin across 32 lanes
        for (int pair = lane_id; pair < nv1sq; pair += WARP_SIZE) {
            const float vx = cvx - L1_RADIUS + d_vel_L1[pair / nv1];
            const float vy = cvy - L1_RADIUS + d_vel_L1[pair % nv1];

            if (vx < vx_lo || vx > vx_hi) continue;
            if (vy < vy_lo || vy > vy_hi) continue;

            float stat = evaluate_trajectory(frames, x0, y0, vx, vy, nx, ny, nf);
            if (stat > local_stat) {
                local_stat = stat;
                local_vx   = vx;
                local_vy   = vy;
            }
        }
    }

    // ---- Warp-merge phase: top-K from 32 lane-local bests ----
    // Uses a `deactivated` flag per lane to prevent re-selection.
    // Invariant: after round r, exactly r lanes have deactivated=true,
    // and slot r of d_topk_L1 has been written.
    bool deactivated = false;

    for (int r = 0; r < TOP_K; r++) {
        // Non-deactivated lanes contribute their stat; deactivated lanes
        // contribute -FLT_MAX so they can never become the winner again.
        float contrib = deactivated ? -FLT_MAX : local_stat;

        // Warp-level max reduction (5 xor-shuffle steps)
        float max_s = contrib;
        max_s = fmaxf(max_s, __shfl_xor_sync(FULL_MASK, max_s, 16));
        max_s = fmaxf(max_s, __shfl_xor_sync(FULL_MASK, max_s, 8));
        max_s = fmaxf(max_s, __shfl_xor_sync(FULL_MASK, max_s,  4));
        max_s = fmaxf(max_s, __shfl_xor_sync(FULL_MASK, max_s,  2));
        max_s = fmaxf(max_s, __shfl_xor_sync(FULL_MASK, max_s,  1));

        // Ballot: non-deactivated lanes that hold the max value
        unsigned int ballot = __ballot_sync(
            FULL_MASK, (!deactivated) && (local_stat == max_s));
        // ballot is always non-zero: with TOP_K=10 and 32 lanes, at most 10
        // lanes are deactivated by round r=9, leaving ≥22 active candidates.
        int winner = __ffs(ballot) - 1;  // lowest lane with the winner stat

        float win_vx = __shfl_sync(FULL_MASK, local_vx, winner);
        float win_vy = __shfl_sync(FULL_MASK, local_vy, winner);

        // Lane r writes its assigned top-K slot
        if (lane_id == r) {
            d_topk_L1_stat[r * n_active + slot] = max_s;
            d_topk_L1_vx  [r * n_active + slot] = win_vx;
            d_topk_L1_vy  [r * n_active + slot] = win_vy;
        }

        // Deactivate the winner for subsequent rounds
        if (lane_id == winner) deactivated = true;
    }
}

// ============================================================
// Kernel 4 (Strategy 3): L2 refinement — one thread per active pixel
// ============================================================
// Reads d_topk_L1 lazily (one candidate at a time → 3 scalars live,
// not TOP_K×3 simultaneously). Tracks only a 3-scalar global best.
// Register budget: ~20 regs → near-peak occupancy.
// ============================================================
__global__ __launch_bounds__(256)
void tbd_L2_pixel(
    const __half* __restrict__ frames,
    const float* __restrict__ d_topk_L1_stat,   // [TOP_K × n_active]
    const float* __restrict__ d_topk_L1_vx,
    const float* __restrict__ d_topk_L1_vy,
    float* __restrict__ d_result_stat,           // [n_active]
    float* __restrict__ d_result_vx,
    float* __restrict__ d_result_vy,
    const int* __restrict__ d_active_pixels,
    int n_active)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_active) return;

    const int pid = d_active_pixels[slot];
    const int nx  = d_nx, ny = d_ny, nf = d_n_frames;
    const int nv2 = d_nv_L2;
    const int x0  = pid % nx;
    const int y0  = pid / nx;

    const float inv_T = 1.0f / (float)(nf - 1);
    const float vx_lo = fmaxf(d_v_min, -(float)x0 * inv_T);
    const float vx_hi = fminf(d_v_max,  (float)(nx - 1 - x0) * inv_T);
    const float vy_lo = fmaxf(d_v_min, -(float)y0 * inv_T);
    const float vy_hi = fminf(d_v_max,  (float)(ny - 1 - y0) * inv_T);

    float best_stat = -FLT_MAX;
    float best_vx   = 0.f;
    float best_vy   = 0.f;

    for (int c = 0; c < TOP_K; c++) {
        // Lazy load: one L1 candidate at a time (only 3 scalars live)
        const float l1s = d_topk_L1_stat[c * n_active + slot];
        if (l1s <= -FLT_MAX) continue;
        const float cvx = d_topk_L1_vx[c * n_active + slot];
        const float cvy = d_topk_L1_vy[c * n_active + slot];

        for (int vi = 0; vi < nv2; vi++) {
            const float vx = cvx - L2_RADIUS + d_vel_L2[vi];
            if (vx < vx_lo || vx > vx_hi) continue;
            for (int vj = 0; vj < nv2; vj++) {
                const float vy = cvy - L2_RADIUS + d_vel_L2[vj];
                if (vy < vy_lo || vy > vy_hi) continue;
                float stat = evaluate_trajectory(frames, x0, y0, vx, vy, nx, ny, nf);
                if (stat > best_stat) {
                    best_stat = stat;
                    best_vx   = vx;
                    best_vy   = vy;
                }
            }
        }
    }

    d_result_stat[slot] = best_stat;
    d_result_vx  [slot] = best_vx;
    d_result_vy  [slot] = best_vy;
}

// ============================================================
// Kernel 5: GPU-side argmax reduction (unchanged)
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

    if (gid < n) { s_stats[tid] = stats[gid]; s_idx[tid] = gid; }
    else         { s_stats[tid] = -FLT_MAX;   s_idx[tid] = -1;  }
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
        block_idx  [blockIdx.x] = s_idx[0];
        int best = s_idx[0];
        block_vx[blockIdx.x] = (best >= 0 && best < n) ? vx_arr[best] : 0.f;
        block_vy[blockIdx.x] = (best >= 0 && best < n) ? vy_arr[best] : 0.f;
    }
}

// ============================================================
// Host helpers
// ============================================================
int build_vel_grid(float* buf, int max_n, float vmin, float vmax, float step) {
    int n = 0;
    for (float v = vmin; v <= vmax + step * 0.4f; v += step) {
        if (n >= max_n) { fprintf(stderr, "Velocity grid overflow (max=%d)\n", max_n); exit(1); }
        buf[n++] = v;
    }
    return n;
}

float* load_frames(const char* filename, int* nx, int* ny, int* nf) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    int header[3];
    if (fread(header, sizeof(int), 3, f) != 3) { fprintf(stderr, "Bad header\n"); exit(1); }
    *nx = header[0]; *ny = header[1]; *nf = header[2];
    size_t np = (size_t)(*nx) * (*ny) * (*nf);
    float* data = (float*)malloc(np * sizeof(float));
    if (fread(data, sizeof(float), np, f) != np) { fprintf(stderr, "Bad frame data\n"); exit(1); }
    fclose(f);
    return data;
}

void load_params(const char* filename, float* v_min, float* v_max, int* n_vel) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    float p[3];
    if (fread(p, sizeof(float), 3, f) != 3) { fprintf(stderr, "Bad params\n"); exit(1); }
    *v_min = p[0]; *v_max = p[1];
    *n_vel = (int)roundf((*v_max - *v_min) / p[2]) + 1;
    fclose(f);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    const char* frame_file = (argc > 1) ? argv[1] : "tbd_frames.bin";
    const char* param_file = (argc > 2) ? argv[2] : "tbd_params.bin";

    printf("================================================================\n");
    printf("  TBD: Strategies 1–4 (Shift-Stack + Spatial Filter + Warp-K)\n");
    printf("================================================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1e9);
    printf("L2:  %.1f MB\n\n", prop.l2CacheSize / 1e6);

    auto t_total = std::chrono::steady_clock::now();

    // ---- Phase 1: Load data ----
    printf("[Phase 1] Loading data...\n");
    auto t0 = std::chrono::steady_clock::now();

    int nx, ny, nf;
    float* h_frames_fp32 = load_frames(frame_file, &nx, &ny, &nf);
    printf("  Frames : %d x %d x %d\n", nx, ny, nf);

    float v_min, v_max; int n_vel;
    load_params(param_file, &v_min, &v_max, &n_vel);
    printf("  Vel    : v_min=%.2f  v_max=%.2f  n_vel=%d\n", v_min, v_max, n_vel);

    double load_ms = std::chrono::duration<double,std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
    printf("  Done   : %.2f s\n\n", load_ms / 1000.0);

    const int n_pixels = nx * ny;
    const int n_active = std::min(n_pixels, N_ACTIVE_MAX);  // Strategy 2 budget

    // ---- Build velocity grids ----
    float h_vel_L0[MAX_VEL_L0], h_vel_L1[MAX_VEL_L1], h_vel_L2[MAX_VEL_L2];
    int nv0 = build_vel_grid(h_vel_L0, MAX_VEL_L0, v_min,  v_max,             L0_STEP);
    int nv1 = build_vel_grid(h_vel_L1, MAX_VEL_L1, 0.0f,   2.0f * L1_RADIUS, L1_STEP);
    int nv2 = build_vel_grid(h_vel_L2, MAX_VEL_L2, 0.0f,   2.0f * L2_RADIUS, L2_STEP);

    long long evals_L0    = (long long)nv0 * nv0;
    long long evals_L1    = (long long)TOP_K * nv1 * nv1;
    long long evals_L2    = (long long)TOP_K * nv2 * nv2;
    long long total_evals = evals_L0 + evals_L1 + evals_L2;
    int nv_brute = 0;
    for (float v = v_min; v <= v_max + 0.02f; v += 0.05f) nv_brute++;
    long long brute_evals = (long long)nv_brute * nv_brute;

    long long total_traj  = (long long)n_pixels * total_evals;
    long long total_reads = total_traj * nf;

    printf("Pyramid:\n");
    printf("  L0: %d×%d = %lld   L1: %d×%d×%d = %lld   L2: %d×%d×%d = %lld\n",
           nv0, nv0, evals_L0,
           TOP_K, nv1, nv1, evals_L1,
           TOP_K, nv2, nv2, evals_L2);
    printf("  Total/pixel: %lld  (brute-force: %lld, %.1fx reduction)\n\n",
           total_evals, brute_evals, (double)brute_evals / total_evals);

    printf("Strategy 2 spatial filter: top-%d / %d pixels for L1+L2  (%.1fx L1+L2 reduction)\n\n",
           n_active, n_pixels, (double)n_pixels / n_active);

    // ---- Upload constants ----
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

    // ---- Allocate device memory ----
    const size_t frame_bytes  = (size_t)n_pixels * nf    * sizeof(__half);
    const size_t stat_bytes   = (size_t)n_pixels          * sizeof(float);
    const size_t topk0_bytes  = (size_t)TOP_K * n_pixels * sizeof(float);  // L0 topk
    const size_t topk1_bytes  = (size_t)TOP_K * n_active * sizeof(float);  // L1 topk (compact)
    const size_t result_bytes = (size_t)n_active          * sizeof(float);  // L2 results

    printf("Device memory:\n");
    printf("  Frames (FP16)   : %.2f GB\n", frame_bytes / 1e9);
    printf("  L0 top-K        : %.2f GB  (%d slots × %d px)\n",
           3.0 * topk0_bytes / 1e9, TOP_K, n_pixels);
    printf("  L1 top-K        : %.2f MB  (%d slots × %d active px)\n",
           3.0 * topk1_bytes / 1e6, TOP_K, n_active);

    __half* d_frames;
    float*  d_H;
    float*  d_topk_L0_stat, *d_topk_L0_vx, *d_topk_L0_vy;
    float*  d_topk_L1_stat, *d_topk_L1_vx, *d_topk_L1_vy;
    float*  d_result_stat,  *d_result_vx,   *d_result_vy;
    int*    d_active_pixels;

    CUDA_CHECK(cudaMalloc(&d_frames,       frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_H,            stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_L0_stat, topk0_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_L0_vx,   topk0_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_L0_vy,   topk0_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_L1_stat, topk1_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_L1_vx,   topk1_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_L1_vy,   topk1_bytes));
    CUDA_CHECK(cudaMalloc(&d_result_stat,  result_bytes));
    CUDA_CHECK(cudaMalloc(&d_result_vx,    result_bytes));
    CUDA_CHECK(cudaMalloc(&d_result_vy,    result_bytes));
    CUDA_CHECK(cudaMalloc(&d_active_pixels, n_active * sizeof(int)));

    // Initialize L0 top-K to -FLT_MAX so any real score beats it
    {
        float* h_init = (float*)malloc(topk0_bytes);
        for (size_t i = 0; i < (size_t)TOP_K * n_pixels; i++) h_init[i] = -FLT_MAX;
        CUDA_CHECK(cudaMemcpy(d_topk_L0_stat, h_init, topk0_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_topk_L0_vx,   h_init, topk0_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_topk_L0_vy,   h_init, topk0_bytes, cudaMemcpyHostToDevice));
        free(h_init);
    }
    printf("  Allocated OK.\n\n");

    // ---- FP32 → FP16 upload ----
    printf("Converting FP32 → FP16 and uploading...\n");
    t0 = std::chrono::steady_clock::now();
    {
        const size_t n_total = (size_t)n_pixels * nf;
        __half* h_fp16 = (__half*)malloc(frame_bytes);
        for (size_t i = 0; i < n_total; i++) h_fp16[i] = __float2half(h_frames_fp32[i]);
        free(h_frames_fp32); h_frames_fp32 = nullptr;
        CUDA_CHECK(cudaMemcpy(d_frames, h_fp16, frame_bytes, cudaMemcpyHostToDevice));
        free(h_fp16);
    }
    double up_ms = std::chrono::duration<double,std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    printf("  Done: %.2f s\n\n", up_ms / 1000.0);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // ================================================================
    // Phase 2a — Strategy 1: L0 shift-and-stack (velocity-outer loop)
    // ================================================================
    // For each of nv0² = 225 velocity pairs: launch shift_and_stack
    // (~6 regs → peak occupancy), then topk_update_L0 (fast-reject).
    // ================================================================
    printf("[Phase 2a] L0 shift-and-stack (%lld velocity pairs)...\n", evals_L0);

    dim3 ss_block(16, 16);   // 256 threads
    dim3 ss_grid((nx + 15) / 16, (ny + 15) / 16);
    const int upd_tpb  = 1024;
    const int upd_grid = (n_pixels + upd_tpb - 1) / upd_tpb;

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int vi = 0; vi < nv0; vi++) {
        for (int vj = 0; vj < nv0; vj++) {
            shift_and_stack<<<ss_grid, ss_block>>>(d_frames, d_H, h_vel_L0[vi], h_vel_L0[vj]);
            topk_update_L0 <<<upd_grid, upd_tpb>>>(
                d_H, d_topk_L0_stat, d_topk_L0_vx, d_topk_L0_vy,
                n_pixels, h_vel_L0[vi], h_vel_L0[vj]);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float l0_ms;
    CUDA_CHECK(cudaEventElapsedTime(&l0_ms, ev0, ev1));
    printf("  Done: %.2f s  (%.1f ms avg per pair)\n\n", l0_ms / 1000.f, l0_ms / evals_L0);

    // ================================================================
    // Phase 2b — Strategy 2: CPU spatial downselection
    // ================================================================
    // Download per-pixel best L0 score (slot 0, slot-major = first n_pixels).
    // CPU nth_element finds top-n_active indices without a full sort.
    // ================================================================
    printf("[Phase 2b] Strategy 2: spatial downselection (top %d / %d pixels)...\n",
           n_active, n_pixels);
    t0 = std::chrono::steady_clock::now();

    std::vector<float> h_l0_best(n_pixels);
    // Slot 0 of the slot-major layout is the first n_pixels floats
    CUDA_CHECK(cudaMemcpy(h_l0_best.data(), d_topk_L0_stat,
                          n_pixels * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int> pixel_indices(n_pixels);
    std::iota(pixel_indices.begin(), pixel_indices.end(), 0);

    // Partial sort: top-n_active by descending L0 score
    std::nth_element(
        pixel_indices.begin(),
        pixel_indices.begin() + n_active,
        pixel_indices.end(),
        [&](int a, int b) { return h_l0_best[a] > h_l0_best[b]; });
    pixel_indices.resize(n_active);

    CUDA_CHECK(cudaMemcpy(d_active_pixels, pixel_indices.data(),
                          n_active * sizeof(int), cudaMemcpyHostToDevice));

    double filter_ms = std::chrono::duration<double,std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
    printf("  Done: %.1f ms  (min L0 score in active set: %.1f)\n\n",
           filter_ms, h_l0_best[pixel_indices.back()]);

    // ================================================================
    // Phase 2c — Strategies 3+4: L1 warp-per-pixel kernel
    // ================================================================
    // 32 threads/pixel, 4 pixels/block.
    // Each lane evaluates a partition of nv1² velocity pairs per L0 candidate.
    // Warp-shuffle top-K merge gathers top-10 from 32 lane bests.
    // ================================================================
    printf("[Phase 2c] L1 refinement: warp-per-pixel on %d pixels...\n", n_active);

    const int l1_tpb   = 128;                          // 4 warps = 4 pixels/block
    const int l1_grid  = (n_active + l1_tpb / WARP_SIZE - 1) / (l1_tpb / WARP_SIZE);

    CUDA_CHECK(cudaEventRecord(ev0));
    tbd_L1_warp<<<l1_grid, l1_tpb>>>(
        d_frames,
        d_topk_L0_stat, d_topk_L0_vx, d_topk_L0_vy,
        d_topk_L1_stat, d_topk_L1_vx, d_topk_L1_vy,
        d_active_pixels, n_active, n_pixels);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float l1_ms;
    CUDA_CHECK(cudaEventElapsedTime(&l1_ms, ev0, ev1));
    printf("  Done: %.2f s\n\n", l1_ms / 1000.f);

    // ================================================================
    // Phase 2d — Strategy 3: L2 per-pixel kernel
    // ================================================================
    // One thread per active pixel. Reads d_topk_L1 lazily (3 regs live).
    // Writes compact d_result arrays indexed by active_slot.
    // ================================================================
    printf("[Phase 2d] L2 refinement: per-pixel on %d pixels...\n", n_active);

    const int l2_tpb  = 256;
    const int l2_grid = (n_active + l2_tpb - 1) / l2_tpb;

    CUDA_CHECK(cudaEventRecord(ev0));
    tbd_L2_pixel<<<l2_grid, l2_tpb>>>(
        d_frames,
        d_topk_L1_stat, d_topk_L1_vx, d_topk_L1_vy,
        d_result_stat, d_result_vx, d_result_vy,
        d_active_pixels, n_active);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float l2_ms;
    CUDA_CHECK(cudaEventElapsedTime(&l2_ms, ev0, ev1));
    printf("  Done: %.2f s\n\n", l2_ms / 1000.f);

    const float total_search_ms = l0_ms + l1_ms + l2_ms;

    // ================================================================
    // Phase 3: GPU argmax reduction on compact results
    // ================================================================
    printf("[Phase 3] GPU argmax on %d candidates...\n", n_active);

    const int red_tpb  = 1024;
    const int red_grid = (n_active + red_tpb - 1) / red_tpb;

    float* d_blk_stats; float* d_blk_vx; float* d_blk_vy; int* d_blk_idx;
    CUDA_CHECK(cudaMalloc(&d_blk_stats, red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vx,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vy,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_idx,   red_grid * sizeof(int)));

    size_t smem = red_tpb * (sizeof(float) + sizeof(int));
    reduce_argmax<<<red_grid, red_tpb, smem>>>(
        d_result_stat, d_result_vx, d_result_vy, n_active,
        d_blk_stats, d_blk_vx, d_blk_vy, d_blk_idx);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_blk_stats(red_grid), h_blk_vx(red_grid), h_blk_vy(red_grid);
    std::vector<int>   h_blk_idx(red_grid);
    CUDA_CHECK(cudaMemcpy(h_blk_stats.data(), d_blk_stats, red_grid*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_vx.data(),    d_blk_vx,    red_grid*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_vy.data(),    d_blk_vy,    red_grid*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_idx.data(),   d_blk_idx,   red_grid*sizeof(int),   cudaMemcpyDeviceToHost));

    float best_stat = -FLT_MAX;
    int   best_slot = -1;
    float det_vx = 0.f, det_vy = 0.f;
    for (int i = 0; i < red_grid; i++) {
        if (h_blk_stats[i] > best_stat) {
            best_stat = h_blk_stats[i];
            best_slot = h_blk_idx[i];   // active_slot index
            det_vx    = h_blk_vx[i];
            det_vy    = h_blk_vy[i];
        }
    }

    // Map active_slot → original pixel index → (x0, y0)
    const int best_pid = (best_slot >= 0 && best_slot < n_active)
                         ? pixel_indices[best_slot] : -1;
    const int det_x0   = (best_pid >= 0) ? best_pid % nx : -1;
    const int det_y0   = (best_pid >= 0) ? best_pid / nx : -1;

    // ---- Results ----
    double total_wall_ms = std::chrono::duration<double,std::milli>(
                               std::chrono::steady_clock::now() - t_total).count();

    printf("\n");
    printf("================================================================\n");
    printf("  RESULTS\n");
    printf("================================================================\n");
    printf("  Detected trajectory:\n");
    printf("    x0   = %d\n",              det_x0);
    printf("    y0   = %d\n",              det_y0);
    printf("    vx   = %.4f px/frame\n",   det_vx);
    printf("    vy   = %.4f px/frame\n",   det_vy);
    printf("    stat = %.1f\n",            best_stat);
    printf("\n");
    printf("  Timing breakdown:\n");
    printf("    Data load + H2D          : %.2f s + %.2f s\n",
           load_ms / 1000.0, up_ms / 1000.0);
    printf("    [S1] L0 shift-and-stack  : %.2f s  (%lld pairs × 2 kernels)\n",
           l0_ms / 1000.f, evals_L0);
    printf("    [S2] CPU spatial filter  : %.1f ms  (%d → %d pixels)\n",
           filter_ms, n_pixels, n_active);
    printf("    [S3+S4] L1 warp-per-pix  : %.2f s\n", l1_ms / 1000.f);
    printf("    [S3] L2 per-pixel        : %.2f s\n", l2_ms / 1000.f);
    printf("    GPU search total         : %.2f s\n", total_search_ms / 1000.f);
    printf("    Wall total               : %.3f s\n", total_wall_ms / 1000.0);
    printf("\n");
    printf("  Throughput:\n");
    printf("    %.2f billion trajectories / s\n",
           total_traj / (total_search_ms * 1e6));
    printf("    Effective BW: %.2f GB/s (FP16 reads)\n",
           total_reads * sizeof(__half) / (total_search_ms * 1e6));
    printf("\n");
    printf("  Pyramid / spatial efficiency:\n");
    printf("    Vel evals/pixel  : %lld  (brute-force: %lld, %.1fx)\n",
           total_evals, brute_evals, (double)brute_evals / total_evals);
    printf("    L1+L2 pixels     : %d / %d  (%.1fx work reduction)\n",
           n_active, n_pixels, (double)n_pixels / n_active);
    printf("================================================================\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_topk_L0_stat)); CUDA_CHECK(cudaFree(d_topk_L0_vx)); CUDA_CHECK(cudaFree(d_topk_L0_vy));
    CUDA_CHECK(cudaFree(d_topk_L1_stat)); CUDA_CHECK(cudaFree(d_topk_L1_vx)); CUDA_CHECK(cudaFree(d_topk_L1_vy));
    CUDA_CHECK(cudaFree(d_result_stat));  CUDA_CHECK(cudaFree(d_result_vx));  CUDA_CHECK(cudaFree(d_result_vy));
    CUDA_CHECK(cudaFree(d_active_pixels));
    CUDA_CHECK(cudaFree(d_blk_stats)); CUDA_CHECK(cudaFree(d_blk_vx));
    CUDA_CHECK(cudaFree(d_blk_vy));    CUDA_CHECK(cudaFree(d_blk_idx));
    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));

    return 0;
}
