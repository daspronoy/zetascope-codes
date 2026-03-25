changes"

1. Memory access coalescing in "evaluate_trajectory"
frames[t * frame_stride + py * nx + px] — each warp lane follows a different (vx, vy) trajectory, so px/py diverge across lanes. This means uncoalesced 128-byte cache lines are loaded from global memory for almost every frame access. On Orin's ~200 GB/s unified bus, this is the dominant bottleneck and neither strategy addresses it.

2. "reduce_argmax" kernel uses shared memory reduction
This is correct but ignores that a simple warp-shuffle reduction (as used in Strategy 4) would be faster and more consistent.

3. Shift-and-Stack (Velocity-Outer Loop)

The most fundamental restructuring. Instead of *pixel-outer, velocity-inner*, flip to *velocity-outer, pixel-inner*.

**Key insight**: For a fixed `(vx, vy)`, every pixel's score is:
```
S(x0, y0) = Σ_t  frame[t][y0 + vy*t][x0 + vx*t]
```
This is a sum of shifted frames — a "shift image". Compute it once for all pixels simultaneously.

```
For each velocity pair (vx, vy):
    GPU kernel: H[y,x] = Σ_t frame[t][y+vy*t][x+vx*t]   // ~5 registers/thread
    GPU kernel: per-pixel topk update → d_topk_L0[pixel][10]  // lives in global mem

Example shift kernel:

```cuda
__global__ void shift_and_accumulate(
    const __half* __restrict__ frames,
    float* __restrict__ H,   // [ny × nx] output
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
        acc += __half2float(frames[(size_t)t * frame_stride + py * nx + px]);
    }
    H[y * nx + x] = acc;
}
```

This kernel uses ~6 registers vs ~100+ in the fused kernel.

---

3. Implement shift-and-stack first — fixes the 64× cache line waste, biggest gain


4. INT8 frames — halves bytes fetched after (1), worthwhile on Orin if still bandwidth-bound