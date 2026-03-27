Changelogs:

v4: Boundary-tightened pixel exclusion
    - Runtimes: kernel time: 2.2 h; wall time: 2.2 h; effective bandwidth: 0.24 GB/s (INT8)
    - pixels within margin = ceil(v_max*(nf-1)) of any edge are skipped at
      all three levels. Those pixels cannot anchor a full trajectory at maximum speed without
      leaving the FOV; their trajectories always return -FLT_MAX regardless of velocity tested.

v3: INT8 quantization + flip shift-and-stack (tbd_v3.cu)
    - Runtimes: kernel time: 2.2 h; wall time: 2.2 h; effective bandwidth: 0.24 GB/s (INT8)
    - SNR (stacked): 24.25
    - INT8 frame storage
     Frames stored as uint8_t (was FP16). Per-frame linear scale:
       scale[t] = max_val_in_frame[t] / 255.0
       dequant : fp32 = u8 * scale[t]
     Scales held in device global memory via __device__ float* g_frame_scale
     (not constant memory, so frame count is unbounded). All warp lanes read
     the same scale[t] per loop iteration → L2 broadcast, no penalty vs
     constant memory. Halves memory traffic vs FP16.
     Assumes non-negative frame values (radar/optical intensity).
 
    - L0 shift-and-stack (velocity-outer, pixel-inner)
     Replaces the old warp-per-pixel L0 search. For each (vx, vy) pair:
       shift_and_accumulate  — one thread per pixel computes
                               H[y,x] = Σ_t frame[t][y+vy*t][x+vx*t]
                               Adjacent threads read adjacent columns
                               → fully coalesced 128-byte cache lines
                               (~6 registers/thread, ~1% vs ~64× waste before)
       topk_update_L0        — one thread per pixel inserts H[pixel]
                               into that pixel's TOP_K candidate list
     Launched nv0² times (one per coarse velocity pair) with init_topk
     initialising d_topk_L0 to -FLT_MAX before the loop.


v2: Multipass kernel split + warp-per-pixel (tbd_v2.cu)
    - Runtimes: kernel time: 6.7 h; wall time: 6.7 h; effective bandwidth: 0.38 GB/s (FP16)
    - MULTI-PASS KERNEL SPLIT
       The single fused kernel (all 3 pyramid levels in one) is split
       into three separate kernels with global-memory intermediates:
         tbd_warp_L0  → writes d_topk_L0[n_pixels × TOP_K]
         tbd_warp_L1  → reads d_topk_L0, writes d_topk_L1
         tbd_warp_L2  → reads d_topk_L1, writes best_stats/vx/vy
       Each kernel holds only its own level's state in registers,
       cutting register pressure roughly in half per kernel vs the
       fused version, improving SM occupancy ~1.5-2x per kernel.
    - WARP-PER-PIXEL
       One warp (32 threads) is assigned to each pixel. The velocity
       search space for that pixel is partitioned across the 32 lanes:
         lane i handles velocity indices i, i+32, i+64, ...
       Each thread tracks only 1 local best (3 scalar registers) rather
       than a topk[TOP_K] array (15 registers). After the inner loop, a
       warp-shuffle top-K extraction writes the results to global memory.
    - Min stacked SNR: ~35



v1: Pyramid search (tbd_v1.cu)
    - Runtimes: kernel time: 12 h; wall time: 12 h; effective bandwidth: 0.21 GB/s (FP16)
    - COARSE-TO-FINE VELOCITY PYRAMID (~25-40x fewer velocity evals)
       - Level 0 (coarse):  step=0.50, 15x15 =  225 hypotheses/pixel
       - Level 1 (medium):  step=0.10, 11x11 =  121 hypotheses/candidate
       - Level 2 (fine):    step=0.05, 5x5   =   25 hypotheses/candidate
    - BOUNDS-AWARE TRAJECTORY PRUNING
       Pre-compute valid velocity range per pixel to skip
       trajectories that immediately exit the FOV.
    - Min stacked SNR: ~35



v0.5: Memory optimizations (tbd_v0.5.cu)
    - Runtimes: kernel time: 9.3 h; wall time: 9.3 h; effective bandwidth: 4.05 GB/s (FP16)
    - FP16 FRAME STORAGE (halves memory footprint & bandwidth)
       Frames are stored as __half on device; each trajectory read
       converts to FP32 for accumulation. Photon counts ~100 are
       well within FP16 range (max representable ~65504).
    - ORIN-SPECIFIC THREAD BLOCK TUNING
       8x16 = 128 threads per block, tuned for 2048 CUDA cores / 16 SMs
       on sm_87. Balances occupancy against register pressure.
    - ROW-CHUNKED KERNEL LAUNCHES
       Splits the NY rows into fixed-size chunks to avoid GPU watchdog
       timeouts on long-running kernels and to enable progress reporting.
    - GPU-SIDE ARGMAX REDUCTION
       The per-pixel best-score array is reduced entirely on the GPU.
       Only ~10K block-level results are copied to host, avoiding a
       large device-to-host transfer for the full n_pixels array.
    - Works for stacked SNR = 8.


v0: Brute-Force method (tbd_v0.cu)
    - Runtimes: kernel time: > 1 day; wall time: > 1 day; effective bandwidth: < 0.1 GB/s (FP32)
    - Basic Stack-and-Shift method over all velocity hypothesis
    - 2D spatial parallelization over (x0, y0)
    - Per-thread local accumulation with register-level tracking
    - Two-stage reduction: per-pixel best → global best
    - Works for stacked SNR = 8.





=============================================================================================================================================================
Benchmark parameters (Jetson Orin)
=============================================================================================================================================================
Resolution: 3072 x 3072 pixels (~9.4 MP)
Number of frames: 300
Brute-force velocity hypothesis baseline: 24025
SNR (stacked, for v1 and v2): 43
SNR (stacked, v3 onwards): 24.25
