# GPU Engineer Interview — Track-Before-Detect

## Overview

Track-Before-Detect (TBD) is used to detect extremely faint objects moving
through image sequences. The object is invisible in any single frame, but
by integrating pixel values along candidate trajectories across many frames,
the signal accumulates and becomes detectable.

This exercise asks you to accelerate the brute-force "shift-and-stack" TBD
algorithm using CUDA and C++.

## Problem Setup

### 1. Generate test data

```bash
python3 tbd_problem.py
```

This creates:
- `tbd_frames.bin` — 16 frames of 256×256 Poisson-noise images with a faint embedded object
- `tbd_params.bin` — velocity search grid parameters
- `tbd_reference.txt` — expected detection result

### 2. Algorithm specification

For every candidate trajectory `(x0, y0, vx, vy)`:

```
T(x0, y0, vx, vy) = Σ_{t=0}^{N-1} frame[t, round(y0 + vy*t), round(x0 + vx*t)]
```

Find the `(x0, y0, vx, vy)` that maximizes `T`.

**Search space:**
- Starting position: all `256 × 256` pixels
- Velocity: `-3.0` to `+3.0` pixels/frame, step `0.25` → `25 × 25 = 625` hypotheses
- **Total: ~41 million trajectories**, each requiring 16 pixel lookups
- Trajectories that go out of bounds at any frame are invalid (skip them)

### 3. Binary data format

**`tbd_frames.bin`:**
```
[int32 nx][int32 ny][int32 n_frames]   // 12-byte header
[float32 × (n_frames × ny × nx)]       // frame data, frame-major row-major
```

**`tbd_params.bin`:**
```
[float32 v_min][float32 v_max][float32 v_step]
```

## Your Task

Implement a CUDA/C++ program that:

1. Reads `tbd_frames.bin` and `tbd_params.bin`
2. Performs the brute-force shift-and-stack search on the GPU
3. Reports the detected `(x0, y0, vx, vy)` and test statistic
4. Reports kernel execution time

### Requirements
- Results must match `tbd_reference.txt`
- Report wall-clock time for the search kernel
- Code should compile with `nvcc`

### Evaluation criteria (in order of importance)
1. **Correctness** — matching detection result
2. **Performance** — GPU kernel throughput
3. **Parallelization strategy** — thread/block mapping, occupancy
4. **Memory optimization** — access patterns, caching, memory hierarchy
5. **Code quality** — readability, error handling, engineering practices

## Reference numbers

| Metric | Value |
|--------|-------|
| Image size | 256 × 256 |
| Frames | 16 |
| Velocity hypotheses | 25 × 25 = 625 |
| Total trajectories | ~41M |
| Total pixel reads | ~655M |
| Python (loops) estimate | ~2-4 hours |
| Target CUDA time | < 100 ms |

Good luck.
