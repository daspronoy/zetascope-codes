# Why the "Smart" Search Lost to Brute Force

---

## Part 1 — The Simple Version

### The Shortcut That Wasn't

When you first look at this problem, a pyramid search seems genius. Instead of checking all
24,025 velocity combinations, you check only 225 rough ones first, find the most promising
ones, then zoom in. That's roughly 14× less work — free speedup, right?

Wrong. And the reason is simple: the shortcut only works if the rough search is actually
useful. In this problem, it isn't.

Here's the intuition. Imagine you're trying to find a single person in a city by scanning
the streets with a low-resolution satellite photo. If your photo is so blurry that everyone
looks the same, your "promising spots" are just random guesses. You've spent time on the
blurry scan and gained nothing. You'd have been better off just walking the streets directly
from the start.

That's exactly what happens here. The coarse level of the pyramid search uses a velocity
step of 0.50 pixels/frame. Across 300 frames, a speed error of 0.50 px/frame means your
predicted position drifts by up to 150 pixels away from the real object by the last frame.
You're sampling completely wrong pixels. The score you get looks just like background noise
— you can't tell a real object from nothing. The "top candidates" coming out of Level 0 are
meaningless, and everything built on top of them (Levels 1 and 2) is wasted work.

There is no free lunch. The pyramid looks like a shortcut, but the shortcut doesn't apply
here.

### Why Brute Force Is Actually Faster

It sounds paradoxical: doing 14× more searches is faster. But once you drop the pyramid and
go brute force, you can also unlock a set of hardware optimizations that were completely
blocked before.

The key ones:

**Storing images at 1 byte per pixel instead of 4.** Instead of carrying full floating-point
pixel values on the GPU, you can compress each frame down to 8-bit integers (0–255) and keep
a single scale factor per frame to convert back. This means the GPU has to read 4× less data
from memory for every trajectory it evaluates. Memory bandwidth is almost always the
bottleneck on GPUs, so 4× less data means nearly 4× faster reads.

**No wasted threads.** The pyramid's refinement stages assigned an entire group of 32 threads
(a "warp") to handle one pixel's search. That's powerful if each pixel has a complicated
search to do, but it comes with heavy overhead — each thread needs to track a lot of state,
which crowds out the number of independent tasks the GPU can juggle at once. The brute force
approach instead assigns one thread to one pixel per velocity pair, which is the simplest
possible assignment. The GPU can pack in far more of these simple tasks simultaneously.

**Skipping the edges automatically.** An object at the very edge of the image can't move
fast without immediately flying off screen, so those starting positions are useless to check.
By computing a border margin upfront and skipping it, you shed a large chunk of work. And
because those edge pixels are gone, you can also remove a costly check inside the inner loop
("is this pixel still on screen?"), letting the GPU run the loop without any interruptions.

Put it all together: brute force reads less data, uses threads more efficiently, and runs a
tighter inner loop. The pyramid reads more data, wastes threads on useless Level 0 results,
and carries extra overhead at every level. Brute force wins — and wins by a lot.

---

## Part 2 — The Technical Version

### The Score Function Is Not Smooth in Velocity Space

The pyramid search is a coarse-to-fine optimizer. Its correctness guarantee is: "the global
optimum in velocity space must lie near a local optimum at the coarser level." This holds
when the objective function is reasonably smooth — when a small change in velocity produces
a small, predictable change in the score.

The TBD score function is not smooth. It is defined as:

    T(x0, y0, vx, vy) = Σ_{t=0}^{N-1}  frame[t,  round(y0 + vy·t),  round(x0 + vx·t)]

The `round()` makes T a step function of (vx, vy). For a point source (signal injected at
one pixel per frame), the contribution at frame t is nonzero only when the rounded position
exactly matches the signal pixel. The score is elevated only for velocities satisfying:

    | (vx - vx_true) · t |  <  0.5   for all t = 0, …, N-1

The binding constraint is at t = N-1:

    | vx - vx_true |  <  1 / (2·(N-1))

For N = 300: the "basin of attraction" in velocity space has half-width ≈ 0.0017 px/frame.
The Level 0 step is 0.50 px/frame — roughly 300× wider than the basin. The probability that
any Level 0 grid point falls inside the basin is ~1/300 per axis, or ~1/90,000 jointly. In
practice it is zero for any given dataset. Level 0 evaluates trajectories that are all 50–150
pixels away from the true trajectory at the last frame. Their scores are drawn from the
background distribution, not elevated by the signal. The Level 0 top-K is a random set, and
all downstream refinement is built on garbage.

This is the "no free lunch" principle applied concretely: coarse-to-fine optimization in
velocity space only accelerates the search when the objective is smooth at the coarse scale.
Here, the objective has correlation length ~1/(N-1) in velocity, and the coarse step is N/2
times larger than that. The pyramid's core assumption is violated by a factor of ~N².

Bilinear interpolation (removing `round()`) would make T differentiable but does not widen
the basin — it only changes the shape of the peak from a step to a triangle. The half-width
becomes 1/(N-1) instead of 1/(2·(N-1)), a factor-of-2 improvement, which is irrelevant
against a 300× coarse step.

### Why Shift-and-Stack Brute Force Is Computationally Superior

**Memory bandwidth.**
The dominant cost is reading frame data. With FP32 frames and the pixel-outer kernel (one
thread per pixel, inner loop over all N_vel² velocity pairs), each thread reads N_vel²·N
frame values from global memory, with an access stride of (vx·t, vy·t) pixels — effectively
random access across frames. Cache hit rate is low.

The velocity-outer shift-and-stack kernel inverts the loop order: for a fixed (vx, vy), one
thread per pixel reads N frames, with adjacent threads (adjacent x) reading adjacent columns
at each frame. This is fully coalesced: all 32 threads in a warp issue a single 128-byte
transaction per frame per row. Memory efficiency is near-optimal.

Switching to INT8 storage (uint8_t + per-frame scale) reduces the frame footprint from 4
B/pixel to 1 B/pixel. Since the access pattern is now coalesced, the GPU's effective memory
bandwidth is fully utilized, and the 4× data reduction translates directly into ~4× throughput
improvement on the frame-read bottleneck.

**Occupancy and register pressure.**
The warp-per-pixel kernels (L1, L2) in the pyramid carry ~35–40 registers per thread: the
local best (stat, vx, vy), the cached L0 top-K array (10 × 3 floats = 30 values, spilled
to local memory if not in registers), and loop indices. At 40 registers/thread, a 128-thread
block consumes 5120 registers, limiting the SM to a small number of concurrent blocks and
reducing occupancy significantly.

The shift-and-stack kernel carries ~6 registers per thread: (x, y, acc, t, px, py). At 6
registers/thread, a 128-thread block consumes 768 registers. The SM can sustain far more
concurrent blocks, keeping the execution units fully fed and hiding memory latency through
warp switching.

**Branch elimination via boundary exclusion.**
For pixel (x, y) with x ∈ [margin, nx−1−margin] and any velocity with |vx| ≤ v_max, the
position at frame t satisfies:

    x + vx·t  ∈  [margin − v_max·(N−1),  (nx−1−margin) + v_max·(N−1)]
             ⊆  [0, nx−1]                 when margin = ⌈v_max·(N−1)⌉

The per-sample bounds check `if (px < 0 || px >= nx || ...)` is therefore dead code for all
inner pixels and all tested velocities. Removing it eliminates branch divergence within warps
and allows the compiler to unroll and pipeline the N-iteration frame loop without guard
conditions — a significant benefit for a 300-iteration loop that dominates runtime.

**Net effect.**
The brute force evaluates ~14× more velocity pairs than the pyramid claims to, but each
evaluation is:
  - 4× cheaper in memory reads (INT8 vs FP32)
  - Better cached (coalesced vs strided access)
  - Executed at higher occupancy (~6 vs ~40 registers/thread)
  - Free of branch divergence in the inner loop

The pyramid's nominal 14× reduction in velocity evaluations is more than erased by these
per-evaluation cost differences, and the pyramid produces wrong answers besides. The
shift-and-stack brute force is the correct algorithm and the faster implementation.
