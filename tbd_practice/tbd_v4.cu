#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// Error checking
// ============================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// Thread-block dimensions (tuned for Orin sm_87)
// BLOCK_X = 32: one full warp per row → coalesced column reads
// BLOCK_Y =  4: 32×4 = 128 threads/block, balances occupancy
//               vs register pressure on 2048-core / 16-SM device
// ============================================================
#define BLOCK_X 32
#define BLOCK_Y  4

// Print search progress after every N velocity pairs
#define REPORT_EVERY 1000

// ============================================================
// Velocity search grid (independent of data generation)
// Matches tbd_problem.py: V_MIN=-3.0, V_MAX=3.0, V_HYP=155
// V_STEP = (V_MAX - V_MIN) / (V_HYP - 1) = 6.0 / 154
// ============================================================
#define SEARCH_V_MIN   (-3.0f)
#define SEARCH_V_MAX   ( 3.0f)
#define SEARCH_V_HYP   155
#define SEARCH_V_STEP  ((SEARCH_V_MAX - SEARCH_V_MIN) / (float)(SEARCH_V_HYP - 1))

// ============================================================
// Constant memory: image geometry + border margin
// ============================================================
__constant__ int d_nx;
__constant__ int d_ny;
__constant__ int d_n_frames;
__constant__ int d_margin;   // = ceil(v_max * (nf-1)): border pixels to skip

// ============================================================
// Per-frame dequantisation scales — device global memory pointer.
// Not constant memory so the frame count is unbounded.
// All warp lanes read the same scale[t] each inner-loop iteration
// → L2 broadcast, no throughput penalty vs constant memory.
// ============================================================
__device__ float* g_frame_scale;

// ============================================================
// Kernel: shift-and-accumulate  (INT8 frames, inner pixels only)
// ============================================================
// For fixed (vx, vy), computes for every inner pixel (x, y):
//
//   H[y, x] = Σ_{t=0}^{nf-1}  u8[t, round(y+vy·t), round(x+vx·t)] · scale[t]
//
// Adjacent threads in x map to adjacent image columns at the same
// (frame, row-offset) → fully coalesced 128-byte cache lines.
//
// No per-sample bounds check: the margin guarantees every
// trajectory from an inner pixel with |vx|, |vy| ≤ v_max stays
// inside [0,nx) × [0,ny) for all t.  Removing the branch
// eliminates divergence and lets the compiler pipeline the loop.
// ============================================================
__global__ void shift_and_accumulate_u8(
    const uint8_t* __restrict__ frames,
    float*         __restrict__ H,
    float vx, float vy)
{
    const int nx       = d_nx,  ny = d_ny, nf = d_n_frames;
    const int margin   = d_margin;
    const int inner_nx = nx - 2 * margin;
    const int inner_ny = ny - 2 * margin;

    const int ix = blockIdx.x * BLOCK_X + threadIdx.x;   // inner-x index
    const int iy = blockIdx.y * BLOCK_Y + threadIdx.y;   // inner-y index

    if (ix >= inner_nx || iy >= inner_ny) return;

    const int x = ix + margin;
    const int y = iy + margin;

    const size_t frame_stride = (size_t)ny * nx;
    float acc = 0.f;

    for (int t = 0; t < nf; t++) {
        // Bounds guaranteed by margin — no branch needed
        const int px = __float2int_rn((float)x + vx * (float)t);
        const int py = __float2int_rn((float)y + vy * (float)t);
        acc += (float)__ldg(&frames[(size_t)t * frame_stride + py * nx + px])
               * g_frame_scale[t];
    }

    H[y * nx + x] = acc;
}

// ============================================================
// Kernel: per-pixel best-1 update
// ============================================================
// Called once per (vx, vy) pair after shift_and_accumulate_u8.
// Replaces the old approach of tracking (vxi, vyi) indices inside
// the search kernel — keeping index integers alongside floats in
// registers hurt occupancy.  Storing float vx/vy directly avoids
// a velocity decode step in the results phase.
// ============================================================
__global__ void update_best(
    const float* __restrict__ H,
    float vx, float vy,
    float* __restrict__ best_stats,
    float* __restrict__ best_vx,
    float* __restrict__ best_vy)
{
    const int nx       = d_nx,  ny = d_ny;
    const int margin   = d_margin;
    const int inner_nx = nx - 2 * margin;
    const int inner_ny = ny - 2 * margin;

    const int ix = blockIdx.x * BLOCK_X + threadIdx.x;
    const int iy = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (ix >= inner_nx || iy >= inner_ny) return;

    const int pid = (iy + margin) * nx + (ix + margin);
    const float s = H[pid];
    if (s > best_stats[pid]) {
        best_stats[pid] = s;
        best_vx[pid]    = vx;
        best_vy[pid]    = vy;
    }
}

// ============================================================
// Kernel: initialise per-pixel best-stat array to -FLT_MAX
// ============================================================
__global__ void init_best(float* best_stats, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) best_stats[i] = -FLT_MAX;
}

// ============================================================
// Kernel: GPU-side warp-shuffle argmax reduction
// ============================================================
// Replaces the shared-memory tree reduction.  Uses warp-shuffle
// intrinsics for the within-warp pass (no shared memory reads),
// then collects one winner per warp into a 32-slot shared buffer
// for the block-level pass.  Only ~red_grid results are copied
// to host, avoiding a full n_pixels device-to-host transfer.
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
    const int tid  = threadIdx.x;
    const int gid  = blockIdx.x * blockDim.x + tid;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float s   = (gid < n) ? stats[gid] : -FLT_MAX;
    int   idx = (gid < n) ? gid        : -1;

    // Warp-level reduce: keep max and its original pixel index
    for (int off = 16; off > 0; off >>= 1) {
        float s2   = __shfl_down_sync(0xffffffff, s,   off);
        int   idx2 = __shfl_down_sync(0xffffffff, idx, off);
        if (s2 > s) { s = s2; idx = idx2; }
    }

    __shared__ float sh_stats[32];
    __shared__ int   sh_idx[32];
    if (lane == 0) { sh_stats[warp] = s; sh_idx[warp] = idx; }
    __syncthreads();

    // Final pass: first warp reduces the per-warp winners
    if (warp == 0) {
        const int nwarps = blockDim.x >> 5;
        s   = (lane < nwarps) ? sh_stats[lane] : -FLT_MAX;
        idx = (lane < nwarps) ? sh_idx[lane]   : -1;
        for (int off = 16; off > 0; off >>= 1) {
            float s2   = __shfl_down_sync(0xffffffff, s,   off);
            int   idx2 = __shfl_down_sync(0xffffffff, idx, off);
            if (s2 > s) { s = s2; idx = idx2; }
        }
        if (lane == 0) {
            block_stats[blockIdx.x] = s;
            block_idx[blockIdx.x]   = idx;
            block_vx[blockIdx.x]    = (idx >= 0) ? vx_arr[idx] : 0.f;
            block_vy[blockIdx.x]    = (idx >= 0) ? vy_arr[idx] : 0.f;
        }
    }
}

// ============================================================
// Host: load FP32 frames from binary file
// Header: 3 × int32 (nx, ny, n_frames)
// Data:   n_frames × ny × nx float32, row-major, frame-major
// ============================================================
float* load_frames(const char* filename, int* nx, int* ny, int* nf)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    int header[3];
    if (fread(header, sizeof(int), 3, f) != 3) {
        fprintf(stderr, "Failed to read header\n"); exit(1);
    }
    *nx = header[0]; *ny = header[1]; *nf = header[2];

    const size_t n = (size_t)(*nx) * (*ny) * (*nf);
    float* data = (float*)malloc(n * sizeof(float));
    if (!data || fread(data, sizeof(float), n, f) != n) {
        fprintf(stderr, "Failed to read frame data\n"); exit(1);
    }
    fclose(f);
    return data;
}


// ============================================================
// Host: quantise FP32 frames → uint8_t with per-frame scale
// ============================================================
// scale[t] = max_val[t] / 255.0
// u8[t,i]  = round( fp32[t,i] * 255 / max_val[t] )   in [0, 255]
// Recovered value: fp32 ≈ u8[t,i] * scale[t]
//
// Halves frame memory vs FP16, quarters vs FP32.
// Assumes non-negative frame values (optical / radar intensity).
// ============================================================
void quantise_frames(
    const float* __restrict__ fp32,
    int nx, int ny, int nf,
    uint8_t** h_u8_out,
    float**   h_scale_out)
{
    const size_t n_pix = (size_t)nx * ny;
    uint8_t* u8    = (uint8_t*)malloc(n_pix * nf);
    float*   scale = (float*)malloc(nf * sizeof(float));

    for (int t = 0; t < nf; t++) {
        const float* src = fp32 + (size_t)t * n_pix;
        uint8_t*     dst = u8  + (size_t)t * n_pix;

        // Per-frame max; floor at 1 to avoid divide-by-zero
        float max_val = 1.f;
        for (size_t i = 0; i < n_pix; i++)
            if (src[i] > max_val) max_val = src[i];

        scale[t] = max_val / 255.f;
        const float inv_s = 255.f / max_val;

        for (size_t i = 0; i < n_pix; i++) {
            // +0.5 before truncation implements round-to-nearest
            const float q = src[i] * inv_s + 0.5f;
            dst[i] = (uint8_t)(q >= 255.5f ? 255 : (unsigned)q);
        }
    }

    *h_u8_out    = u8;
    *h_scale_out = scale;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv)
{
    const char* frame_file = (argc > 1) ? argv[1] : "tbd_frames.bin";

    printf("================================================================\n");
    printf("  TBD Brute-Force — INT8 Shift-and-Stack\n");
    printf("  Opts: INT8 quant · coalesced reads · boundary exclusion\n");
    printf("        GPU warp-shuffle argmax · 128-thread blocks (Orin sm_87)\n");
    printf("================================================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB memory)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1e9);
    printf("L2 cache: %.1f MB\n\n", prop.l2CacheSize / 1e6);

    const auto t_wall_start = std::chrono::steady_clock::now();

    // ---- [1] Load -----------------------------------------------
    printf("[1] Loading data...\n");

    int nx, ny, nf;
    float* h_fp32 = load_frames(frame_file, &nx, &ny, &nf);
    printf("    Frames : %d x %d x %d  (%.2f GB FP32)\n",
           nx, ny, nf, (double)nx * ny * nf * 4 / 1e9);

    const float v_min  = SEARCH_V_MIN;
    const float v_max  = SEARCH_V_MAX;
    const float v_step = SEARCH_V_STEP;
    const int   n_vel  = SEARCH_V_HYP;
    printf("    Vel    : v_min=%.4f  v_max=%.4f  step=%.6f  n_vel=%d\n\n",
           v_min, v_max, v_step, n_vel);

    // ---- [2] INT8 quantisation ----------------------------------
    // Frames converted from FP32 to uint8_t with a per-frame linear
    // scale factor.  The scale is stored on the device and applied
    // during accumulation: fp32 ≈ u8 * scale[t].
    printf("[2] Quantising to INT8...\n");

    uint8_t* h_u8;
    float*   h_scale;
    quantise_frames(h_fp32, nx, ny, nf, &h_u8, &h_scale);
    free(h_fp32); h_fp32 = nullptr;

    printf("    Frame store : %.2f GB INT8  (4x reduction vs FP32)\n\n",
           (double)nx * ny * nf / 1e9);

    // ---- [3] Boundary-tightened pixel exclusion -----------------
    // Pixels within margin = ceil(v_max*(nf-1)) of any edge cannot
    // anchor a full-length trajectory at maximum speed without
    // leaving the FOV — their trajectories always yield -FLT_MAX.
    // Skipping them entirely also lets us remove the per-sample
    // bounds check inside the inner frame loop for inner pixels,
    // eliminating branch divergence and improving pipelining.
    const int margin   = (int)ceilf(v_max * (float)(nf - 1));
    const int inner_nx = nx - 2 * margin;
    const int inner_ny = ny - 2 * margin;

    if (inner_nx <= 0 || inner_ny <= 0) {
        fprintf(stderr,
                "ERROR: margin (%d px) >= half image size. "
                "v_max too large for this image/frame count.\n", margin);
        exit(1);
    }

    printf("[3] Boundary exclusion\n");
    printf("    Margin     : %d px  (= ceil(%.4f * %d))\n",
           margin, v_max, nf - 1);
    printf("    Inner grid : %d x %d = %d pixels  (%.1f%% of full image)\n\n",
           inner_nx, inner_ny, inner_nx * inner_ny,
           100.0 * inner_nx * inner_ny / ((double)nx * ny));

    const int n_pixels = nx * ny;

    // ---- Upload constants to device -----------------------------
    CUDA_CHECK(cudaMemcpyToSymbol(d_nx,       &nx,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ny,       &ny,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_frames, &nf,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_margin,   &margin, sizeof(int)));

    // ---- [4] Allocate + upload device memory --------------------
    uint8_t* d_frames;       // INT8 frame cube  [nf × ny × nx]
    float*   d_scale;        // per-frame scale  [nf]
    float*   d_H;            // shift-and-stack intermediate [ny × nx]
    float*   d_best_stats;   // per-pixel best stat          [ny × nx]
    float*   d_best_vx;      // per-pixel best vx            [ny × nx]
    float*   d_best_vy;      // per-pixel best vy            [ny × nx]

    const size_t frame_bytes = (size_t)nx * ny * nf;           // 1 B/pixel (INT8)
    const size_t stat_bytes  = (size_t)n_pixels * sizeof(float);

    printf("[4] Allocating GPU memory...\n");
    printf("    INT8 frames : %.2f GB\n", frame_bytes / 1e9);
    printf("    Float arrays: %.2f MB  (H + best_stats + best_vx + best_vy)\n",
           4.0 * stat_bytes / 1e6);

    CUDA_CHECK(cudaMalloc(&d_frames,     frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_scale,      (size_t)nf * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H,          stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_stats, stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vx,    stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_vy,    stat_bytes));

    CUDA_CHECK(cudaMemcpy(d_frames, h_u8,    frame_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale,  h_scale, (size_t)nf * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_u8);    h_u8    = nullptr;
    free(h_scale); h_scale = nullptr;

    // Point the __device__ global at the device-side scale array
    CUDA_CHECK(cudaMemcpyToSymbol(g_frame_scale, &d_scale, sizeof(float*)));

    // Initialise per-pixel tracking (vx/vy zeroed; stats to -FLT_MAX)
    CUDA_CHECK(cudaMemset(d_best_vx, 0, stat_bytes));
    CUDA_CHECK(cudaMemset(d_best_vy, 0, stat_bytes));
    {
        const int ib = 256;
        const int ig = (n_pixels + ib - 1) / ib;
        init_best<<<ig, ib>>>(d_best_stats, n_pixels);
        CUDA_CHECK(cudaGetLastError());
    }
    printf("    Done.\n\n");

    // ---- [5] Brute-force velocity-outer search ------------------
    // For each (vx, vy) pair:
    //   1. shift_and_accumulate_u8 — one thread per inner pixel,
    //      fully coalesced reads, no bounds check
    //   2. update_best             — one thread per inner pixel,
    //      updates per-pixel best (stat, vx, vy)
    // Pairs are enumerated as a flat double loop so launch overhead
    // is bounded and progress can be reported mid-search.
    const long long total_pairs = (long long)n_vel * n_vel;
    const long long inner_pix   = (long long)inner_nx * inner_ny;

    printf("[5] Brute-force shift-and-stack\n");
    printf("    Velocity pairs : %d x %d = %lld\n", n_vel, n_vel, total_pairs);
    printf("    Inner pixels   : %d x %d = %lld\n", inner_nx, inner_ny, inner_pix);
    printf("    Total traj     : %.3f billion\n\n",
           (double)total_pairs * inner_pix / 1e9);

    // 32×4 = 128-thread blocks: coalesced in x, tuned for Orin sm_87
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((inner_nx + BLOCK_X - 1) / BLOCK_X,
                    (inner_ny + BLOCK_Y - 1) / BLOCK_Y);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    const auto  t_search  = std::chrono::steady_clock::now();
    long long   pairs_done = 0;

    for (int vi = 0; vi < n_vel; vi++) {
        for (int vj = 0; vj < n_vel; vj++) {
            const float vx = v_min + vi * v_step;
            const float vy = v_min + vj * v_step;

            shift_and_accumulate_u8<<<grid, block>>>(d_frames, d_H, vx, vy);
            update_best<<<grid, block>>>(d_H, vx, vy,
                                         d_best_stats, d_best_vx, d_best_vy);
            ++pairs_done;

            if (pairs_done % REPORT_EVERY == 0 || pairs_done == total_pairs) {
                CUDA_CHECK(cudaDeviceSynchronize());
                const double pct = 100.0 * pairs_done / total_pairs;
                const double el  = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_search).count();
                const double eta = (pairs_done < total_pairs)
                    ? el * (total_pairs - pairs_done) / pairs_done : 0.0;
                printf("\r  %lld/%lld pairs  (%.1f%%)  %.1f s  ETA %.0f s    ",
                       pairs_done, total_pairs, pct, el, eta);
                fflush(stdout);
            }
        }
    }
    printf("\n");

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float search_ms;
    CUDA_CHECK(cudaEventElapsedTime(&search_ms, ev0, ev1));
    printf("\n  Search complete: %.2f s\n\n", search_ms / 1000.f);

    // ---- [6] GPU warp-shuffle argmax reduction ------------------
    // Only ~red_grid (≈9K for 9.4MP image) block-level results are
    // transferred to host for the final sequential pass, rather than
    // the full n_pixels (≈9.4M) best_stats array.
    printf("[6] GPU argmax reduction...\n");

    const int red_block = 1024;
    const int red_grid  = (n_pixels + red_block - 1) / red_block;

    float* d_blk_stats;
    float* d_blk_vx;
    float* d_blk_vy;
    int*   d_blk_idx;
    CUDA_CHECK(cudaMalloc(&d_blk_stats, red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vx,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_vy,    red_grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blk_idx,   red_grid * sizeof(int)));

    reduce_argmax<<<red_grid, red_block>>>(
        d_best_stats, d_best_vx, d_best_vy, n_pixels,
        d_blk_stats, d_blk_vx, d_blk_vy, d_blk_idx);
    CUDA_CHECK(cudaGetLastError());

    float* h_blk_stats = (float*)malloc(red_grid * sizeof(float));
    float* h_blk_vx    = (float*)malloc(red_grid * sizeof(float));
    float* h_blk_vy    = (float*)malloc(red_grid * sizeof(float));
    int*   h_blk_idx   = (int*)  malloc(red_grid * sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_blk_stats, d_blk_stats,
                          red_grid * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_vx,   d_blk_vx,
                          red_grid * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_vy,   d_blk_vy,
                          red_grid * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blk_idx,  d_blk_idx,
                          red_grid * sizeof(int),   cudaMemcpyDeviceToHost));

    float best_stat = -FLT_MAX;
    int   best_idx  = -1;
    float det_vx = 0.f, det_vy = 0.f;
    for (int i = 0; i < red_grid; i++) {
        if (h_blk_stats[i] > best_stat) {
            best_stat = h_blk_stats[i];
            best_idx  = h_blk_idx[i];
            det_vx    = h_blk_vx[i];
            det_vy    = h_blk_vy[i];
        }
    }

    const int det_x0 = (best_idx >= 0) ? (best_idx % nx) : -1;
    const int det_y0 = (best_idx >= 0) ? (best_idx / nx) : -1;

    // ---- Results ------------------------------------------------
    const double total_wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_wall_start).count();
    const long long total_reads = total_pairs * inner_pix * nf;

    printf("\n================================================================\n");
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
    printf("    Search kernel time : %.2f s\n", search_ms / 1000.f);
    printf("    Total wall time    : %.3f s\n", total_wall_s);
    printf("\n");
    printf("  Throughput:\n");
    printf("    %.2f B trajectories/s\n",
           (double)(total_pairs * inner_pix) / (search_ms * 1e6));
    printf("    Effective BW: %.2f GB/s  (INT8 frame reads)\n",
           (double)total_reads / (search_ms * 1e6));
    printf("================================================================\n");

    // ---- Cleanup ------------------------------------------------
    free(h_blk_stats); free(h_blk_vx); free(h_blk_vy); free(h_blk_idx);
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_best_stats));
    CUDA_CHECK(cudaFree(d_best_vx));
    CUDA_CHECK(cudaFree(d_best_vy));
    CUDA_CHECK(cudaFree(d_blk_stats));
    CUDA_CHECK(cudaFree(d_blk_vx));
    CUDA_CHECK(cudaFree(d_blk_vy));
    CUDA_CHECK(cudaFree(d_blk_idx));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    return 0;
}
