# Project Notes — Track-Before-Detect (TBD) GPU Exercise

These notes summarize the repository `tbd_practice` and provide detailed,
section-by-section descriptions of each source and support file used for the
Track-Before-Detect GPU interview problem. Use these notes as a quick
reference while implementing, debugging, or benchmarking the CUDA solution.

---

## Summary

- Purpose: port a brute-force "shift-and-stack" Track-Before-Detect algorithm
	to CUDA/C++. The goal is to find a moving object's trajectory by summing
	pixel values along candidate trajectories over multiple frames.
- Primary files:
	- `tbd_problem.py` — data generator and NumPy reference implementation
	- `tbd_frames.bin`, `tbd_params.bin` — binary inputs produced by Python
	- `tbd_solution.cu` — CUDA/C++ reference solution (search + reduction)
	- `tbd_reference.txt` — ground-truth detection output used for verification
	- `Makefile` — build / run helpers

---

## File: `tbd_practice/README.md`

- High-level problem statement and workflow for the interview task.
- Key algorithm: for each candidate `(x0,y0,vx,vy)` compute
	`T(x0,y0,vx,vy) = sum_{t=0}^{N-1} frame[t, round(y0+vy*t), round(x0+vx*t)]`.
- Search domain: every pixel (256×256) × velocity grid (vx,vy in [-3,3],
	step 0.25 → 25×25 hypotheses) → ~41M trajectories total for N=16 frames.
- Binary formats documented exactly (header + float32 frame data, and three
	float32 velocity params). Must be matched precisely by the C/CUDA reader.
- Requirements: correctness (match `tbd_reference.txt`), report kernel time,
	compile with `nvcc`, aim for high throughput.

Notes:
- The README contains reference metrics and guidance (e.g., suggested
	target kernel time <100 ms) and explains invalidating trajectories that
	go out of bounds.
- Section-by-section meaning:
	- `Overview` explains the core TBD intuition: the object is too weak to see
		in one frame, so detection comes from coherent temporal integration.
	- `Generate test data` defines the first expected workflow step and makes it
		clear that the CUDA program is not responsible for generating inputs.
	- `Algorithm specification` is the most important correctness section because
		it defines the exact objective function and the invalid-trajectory rule.
	- `Binary data format` is the contract between Python and CUDA/C++.
	- `Your Task` and `Requirements` explain the deliverable and what must be
		printed at runtime.
	- `Evaluation criteria` implies that a fast but numerically wrong answer is
		not acceptable; correctness is explicitly ranked above performance.
	- `Reference numbers` translate the algorithm into scale: ~41M trajectories
		and ~655M pixel reads, which motivates GPU acceleration.

---

## File: `tbd_practice/Makefile`

- Purpose: build and run the CUDA solution and generate test data.
- Important variables:
	- `NVCCFLAGS` includes `-O3`, `--use_fast_math`, and a GPU architecture
		`-arch=sm_86` (adjust for your GPU; e.g., `sm_80`, `sm_86`, `sm_89`).
- Targets:
	- `all`: builds `tbd_solution` using `nvcc`.
	- `data`: runs `python3 tbd_problem.py` to produce `tbd_frames.bin` and
		`tbd_params.bin`.
	- `run`: generates data and runs the compiled CUDA binary.
	- `clean`: removes generated binaries and data files.

Notes:
- On Windows, the `run` and `data` targets assume a Unix-like shell; use
	WSL or adapt commands to PowerShell as needed.
- Additional details:
	- `NVCC = nvcc` assumes the CUDA toolkit is on `PATH`.
	- `TARGET = tbd_solution` means the produced executable is named without a
		platform-specific extension in the Makefile, which is common in Unix-like
		environments but may become `tbd_solution.exe` on Windows builds.
	- The compile rule depends only on `tbd_solution.cu`, so header changes
		would not trigger rebuilds unless dependencies are added manually.
	- `run: data $(TARGET)` encodes the intended end-to-end workflow: generate
		fresh data, compile the solver, then execute the solver on those files.
	- `clean` removes both build artifacts and generated data, returning the
		folder to a near-pristine state.

---

## File: `tbd_practice/tbd_problem.py`

Purpose:
- Create reproducible synthetic test data and produce a fast NumPy-based
	reference result used to validate the CUDA implementation.

Imports and what they are used for:
- `numpy as np`: array creation, Poisson sampling, vectorized coordinate
	math, masking, and reduction.
- `time`: elapsed-time reporting for brute-force and vectorized search.
- `struct`: binary packing of headers and parameter files.
- `os`: reporting file sizes after saving binaries.
- `sys`: checking whether `--bruteforce` was passed on the command line.

Key parameters (fixed in the script):
- Image size: `NX=256`, `NY=256`.
- Frames: `N_FRAMES=16`.
- Background: Poisson mean `BG_MEAN=100.0`.
- Object per-frame signal: `OBJ_SIGNAL=20.0`.
- Ground-truth trajectory: `TRUE_X0=128.0`, `TRUE_Y0=100.0`,
	`TRUE_VX=1.5`, `TRUE_VY=-0.75`.
- Velocity grid: `V_MIN=-3.0`, `V_MAX=3.0`, `V_STEP=0.25`.

Why these constants matter:
- `BG_MEAN` and `OBJ_SIGNAL` determine the SNR regime. The comments show the
	object is only marginally visible per frame but becomes obvious after
	stacking 16 frames.
- `SEED=42` guarantees reproducible data, which is essential for comparing a
	CUDA result against a fixed reference file.
- The ground-truth trajectory stays comfortably in bounds over 16 frames,
	which ensures the correct answer is a valid trajectory and not masked out.

Functions and behavior:
- `generate_data()`:
	- Creates a `(N_FRAMES, NY, NX)` float32 array.
	- For each frame, draws Poisson background samples with mean `BG_MEAN`.
	- Injects the object signal at the (rounded) ground-truth position using
		an additional Poisson draw of mean `OBJ_SIGNAL`.
	- Returns frames (frame-major ordering).
	- The object injection is additive, so the target pixel contains background
		plus object counts, not object counts alone.
- `save_binary(frames)`:
	- Writes a header of three 32-bit integers `(NX, NY, N_FRAMES)` followed by
		raw `float32` frame bytes in frame-major, row-major layout.
	- This exact ordering and sizes must be matched by the C++ reader.
	- The function explicitly casts to `float32` before writing, even if the
		input array is already float32, which makes the file format robust.
- `save_params()`:
	- Writes three `float32` values `(V_MIN, V_MAX, V_STEP)` in binary.
- `vectorized_tbd(frames)`:
	- Vectorized NumPy shift-and-stack; iterates velocity pairs but computes
		statistics for all `(x0,y0)` simultaneously using broadcasting.
	- Masks invalid trajectories (any frame out-of-bounds) and sets them to
		a large negative sentinel so they are not chosen.
	- Returns detected `(x0,y0,vx,vy,stat)` and elapsed time; used to write
		`tbd_reference.txt`.
	- `x0_grid` has shape `(nx,)` and `y0_grid` has shape `(ny,)`; inside the
		frame loop these become 2D through broadcasting.
	- `px` and `py` are computed independently for one axis at a time, then
		combined into a 2D validity mask using an outer-product style broadcast.
	- Invalid coordinates are temporarily clamped with `np.clip` so NumPy can
		still gather safely; correctness is preserved because `all_valid` later
		forces those trajectories to `-1e30`.
	- `frames[t][np.ix_(py_safe, px_safe)]` gathers every candidate pixel for
		the current frame at once, producing a full `(ny, nx)` contribution map.
	- After all frames are accumulated, `np.argmax` finds the best starting
		pixel for the current velocity pair, and the outer loops keep the best
		velocity pair seen so far.
- `brute_force_tbd(frames, max_rows=None)`:
	- Pure Python nested loops implementation; extremely slow for the full
		image, provided only for debugging or tiny patches.
	- It is the clearest statement of the intended algorithm because it literally
		loops over `y0`, `x0`, `vx`, `vy`, and frame index `t`.
	- It prints progress every 16 rows so the user can estimate total runtime.
	- The return value is the same shape as the vectorized version, making it
		easy to compare implementations.

Important implementation notes to match in CUDA:
- Rounding semantics: Python `round()` is used when injecting the object and
	when sampling coordinates in the reference; replicate equivalent rounding
	in CUDA. In the actual CUDA file, this is implemented with
	`__float2int_rn(...)`, which rounds to nearest-even and matches the
	NumPy/Python reference behavior for this problem. `roundf()` would not be
	a perfect drop-in replacement because it uses half-away-from-zero behavior.
- Data layout: frames written are `(t, y, x)` with `x` fastest. Device code
	must index as `index = t*(nx*ny) + y*nx + x`.

Output:
- `tbd_reference.txt` is produced with lines: `x0=...`, `y0=...`, `vx=...`,
	`vy=...`, `statistic=...`.

Main program flow:
- Prints a banner describing the exercise.
- Generates fresh synthetic data.
- Saves binary data and prints the true injected trajectory and SNR summary.
- Computes the reference answer with `vectorized_tbd` and prints detection
	accuracy relative to ground truth.
- Writes `tbd_reference.txt` for downstream validation.
- Optionally benchmarks a tiny slice of the pure Python brute-force code if
	`--bruteforce` is passed.
- Ends with a reminder that the CUDA solution should now be implemented and
	matched against the generated files.

---

## File: `tbd_practice/tbd_solution.cu` (reference CUDA implementation)

Overview:
- CUDA/C++ implementation that implements the brute-force shift-and-stack on
	the GPU with the following design features:
	- 2D thread mapping: each GPU thread handles one starting pixel `(x0,y0)`.
	- Small velocity grid stored in device constant memory for broadcast reads.
	- Use of read-only caching (`__ldg()` or `texture` / L1 caching) for frame
		reads to improve throughput.
	- Per-thread accumulation and a per-pixel best-tracking strategy.
	- Two-stage reduction: first per-pixel bests are written, then a block-wise
		reduction finds per-block winners, and finally a small reduction yields
		the global best.

Headers and support code:
- `<cstdio>`, `<cstdlib>`, and `<cstring>` provide C-style I/O and utility
	functions.
- `<cmath>` and `<cfloat>` support rounding and sentinel constants like
	`-FLT_MAX`.
- `<cuda_runtime.h>` exposes the CUDA runtime API used for memory allocation,
	device property queries, event timing, and kernel launches.
- `CUDA_CHECK(...)` wraps CUDA API calls, prints file/line diagnostics on
	failure, and terminates immediately. This keeps the code short while still
	being explicit about runtime failures.

Key constants & device-stored config:
- `MAX_VEL` and `MAX_FRAMES`: compile-time maxima for constant arrays.
- `__constant__ float d_vel[MAX_VEL]`: velocity values stored in constant
	memory for fast broadcast.
- `__constant__ int d_n_vel, d_n_frames, d_nx, d_ny`: small ints stored in
	constant memory so kernels can read them cheaply.

Why constant memory is used:
- The velocity grid and dimensions are small and read by many threads.
- Constant memory is a good fit for broadcast-style access, especially when
	many threads in a warp read the same velocity entry at the same time.
- Storing `nx`, `ny`, `nf`, and `n_vel` in constant memory avoids repeated
	passing of tiny scalars through kernel arguments.

Primary kernels:
- `tbd_search_kernel(frames, best_stats, best_vxi, best_vyi)`:
	- Each thread handles one `(x0,y0)` and loops across all `(vx,vy)`.
	- For each `(vx,vy)`, thread computes per-frame positions `px,py` with
		nearest-integer rounding and checks bounds. If any frame is out of
		bounds, the trajectory is skipped.
	- The actual implementation uses `#pragma unroll 4` on the frame loop to
		encourage partial unrolling across the 16-frame accumulation.
	- Sums pixel values across frames for valid trajectories using read-only
		loads, tracks the best statistic and the indices of the best velocity.
	- Writes one `best_stats[idx]` and `best_vxi/ best_vyi` per thread.
	- Thread mapping details:
		- `x0 = blockIdx.x * blockDim.x + threadIdx.x`
		- `y0 = blockIdx.y * blockDim.y + threadIdx.y`
		- Threads outside the image bounds return immediately.
	- Local variables:
		- `frame_stride = ny * nx` is the number of elements per frame.
		- `local_best` starts at `-FLT_MAX` so any valid trajectory beats it.
		- `local_vxi` and `local_vyi` record which velocity pair produced the
			best statistic for that pixel.
	- Inner-loop logic:
		- `vx = d_vel[vi]` and `vy = d_vel[vj]` are read from constant memory.
		- `stat` accumulates the sum for one candidate trajectory.
		- `valid` tracks whether every predicted position stayed in bounds.
		- If an out-of-bounds sample is detected, the frame loop breaks early,
			which avoids wasted memory reads.
	- Memory behavior:
		- Reads from `frames` are irregular because each trajectory moves across
			the image, so spatial coalescing is limited.
		- `__ldg` routes loads through the read-only path, which can help when
			multiple nearby trajectories reuse cached pixels.
		- Output writes are simple and contiguous in `x`, so the result arrays are
			much more cache- and bandwidth-friendly than the input reads.

- `reduce_max_kernel(stats, vxi, vyi, n_pixels, block_results)`:
	- 1D reduction kernel where each block loads a contiguous chunk of the
		`stats` array into shared memory along with corresponding indices and
		velocity indices.
	- Performs a tree-reduction in shared memory to find the block-local
		maximum; writes a `TBDResult` (stat, pixel_idx, vxi, vyi) per block.
	- Host then copies `block_results` and does a final reduction to get the
		global maximum (or launches a final tiny kernel).
	- Shared memory layout details:
		- One raw `extern __shared__` buffer is partitioned manually.
		- `s_stat` holds the block's candidate statistics.
		- `s_idx`, `s_vxi`, and `s_vyi` hold the matching metadata for each stat.
		- This manual packing avoids multiple shared allocations and keeps all
			reduction state adjacent.
	- Sentinel behavior:
		- Threads whose global index falls beyond `n_pixels` store `-FLT_MAX`
			so they cannot win the reduction.
	- Reduction pattern:
		- Each iteration halves the active lane count.
		- When a higher statistic is found in the upper half, the lower-half slot
			copies all associated metadata, not just the statistic.
		- Thread 0 writes the final block winner to `block_results[blockIdx.x]`.

Host-side flow summarized:
- Read the binary frames and params using `load_frames()` and `load_params()`.
- Build the host velocity grid `h_vel[]` with inclusive upper bound logic
	(`v <= v_max + 0.5 * v_step`) to match the Python generator.
- Upload `h_vel[]` and small constants to device constant memory
	(`cudaMemcpyToSymbol`).
- Allocate `d_frames`, `d_best_stats`, `d_best_vxi`, `d_best_vyi`.
- Copy frames to device memory.
- Configure kernel launch: `block(16,16)` and grid sized to cover `nx × ny`.
- Warmup kernel launch, then timed kernel launch with CUDA events to get
	an accurate `Search kernel time` in milliseconds.
- Run `reduce_max_kernel` with `RED_BLOCK=256`, allocate `d_block_results`.
- Copy `d_block_results` to host and perform final CPU reduction; compute
	`(x0,y0)` from `pixel_idx` and map `vxi/vyi` indices to velocity values.
- Print final detection, reduction time, total GPU time, kernel throughput,
	and effective read bandwidth, then cleanup.

Detailed host execution steps:
- `main(int argc, char** argv)` accepts optional file paths, defaulting to
	`tbd_frames.bin` and `tbd_params.bin`.
- GPU info is printed using `cudaGetDeviceProperties`, including device name,
	compute capability, number of SMs, and total global memory.
- `load_frames()`:
	- Opens the binary file in `rb` mode.
	- Reads three integers into `header` and assigns `nx`, `ny`, `nf`.
	- Allocates host memory with `malloc` for the entire frame stack.
	- Reads all float32 frame values and returns the pointer.
- `load_params()`:
	- Opens the params file and reads exactly three floats.
	- Returns velocity min, max, and step through output pointers.
- Velocity grid construction:
	- A host array `h_vel[MAX_VEL]` is filled from `v_min` to `v_max`.
	- The code guards against exceeding `MAX_VEL`.
	- `n_vel` is later copied into constant memory for device-side loops.
- Diagnostic counters:
	- `n_pixels = nx * ny`
	- `total_traj = n_pixels * n_vel * n_vel`
	- `total_reads = total_traj * nf`
	- These are printed so the user can relate runtime to total work.
- Device allocation:
	- `d_frames` stores the full input cube.
	- `d_best_stats`, `d_best_vxi`, and `d_best_vyi` store one best result per
		starting pixel.
- Launch configuration:
	- `dim3 block(16,16)` gives 256 threads per block.
	- `grid` is computed by ceiling-dividing image dimensions by block size.
	- This maps naturally to the 2D image domain.
- Timing flow:
	- Two CUDA events (`start`, `stop`) are created.
	- The search kernel is launched once as a warmup and synchronized.
	- The timed launch records `start`, launches the kernel, records `stop`,
		then synchronizes on `stop` before computing elapsed time.
	- The reduction stage is timed separately using the same events.
- Final decoding:
	- `pixel_idx` is converted back to `(x0,y0)` using modulo and division.
	- `vxi` and `vyi` index into the host velocity array `h_vel`.
	- The printed statistic is the best accumulated score for the winning
		trajectory.
- Performance metrics:
	- `Kernel throughput` is reported in billions of trajectories per second.
	- `Effective bandwidth` estimates read bandwidth from total pixel reads and
		kernel time only; it is a derived metric rather than a hardware counter.

Correctness and gotchas:
- Precisely match binary layout and rounding used by `tbd_problem.py`.
- Use identical inclusive velocity grid generation to avoid off-by-one in
	`n_vel` between host and device.
- Ensure proper bounds checking per-frame and skip invalid trajectories
	rather than clamping.
- The reduction timing in the current CUDA file includes both the reduction
	kernel launch and the device-to-host copy of block results before the final
	CPU-side reduction.
- Confirm that `__ldg()` or equivalent read-only caching is available and
	helps performance on your architecture; otherwise standard reads will be
	functionally correct but may be slower.
- The code does not compare against `tbd_reference.txt` automatically; the
	user must visually or script-wise compare the printed result to the file.
- The code checks CUDA API failures through `CUDA_CHECK`, but it does not call
	`cudaGetLastError()` immediately after kernel launches; launch failures are
	still likely to surface at the next synchronized CUDA operation.

---

## File: `tbd_practice/tbd_reference.txt`

- Contains the NumPy reference detection produced by `tbd_problem.py`.
- Exact contents (used for verification):
	- `x0=128`
	- `y0=100`
	- `vx=1.5000`
	- `vy=-0.7500`
	- `statistic=1881.000000`

Notes:
- The `statistic` value is the sum of pixel values along the true trajectory
	(including background + object signal draws). If your CUDA result differs
	in `statistic`, rounding indexing or accumulation logic is likely incorrect.
- The formatting is intentionally plain text and line-based, making it easy to
	parse from a script if automated validation is added later.
- `vx` and `vy` are written with four decimal places, while `statistic` is
	written with six. The CUDA program prints fewer decimals for some values,
	so comparison should be numeric rather than purely textual.

---

## File: `tbd_practice/notes.md` (this file)

- Purpose: working notes, benchmarks, to-dos and observations. This file is
	intentionally writable so you can append GPU profiling results, tuning
	experiments, and environment-specific notes (e.g., `nvcc` version,
	compute capability, host OS quirks).

Suggested content to add during development:
- GPU and driver info used for testing.
- `nvcc` invocation flags and rationale (why `--use_fast_math`, chosen
	architecture `-arch=sm_xx`).
- Per-kernel profiling results (occupancy, shared mem usage, register
	usage, achieved GFLOPS/memory bandwidth).
- Micro-benchmarks: time spent generating velocity grid, H2D copy time,
	kernel runtime, reduction runtime.

How to use this file effectively:
- Keep a short log of each optimization attempt and its measured effect.
- Record whether a change affected correctness, speed, or both.
- Note any environment assumptions, especially Windows vs WSL build quirks.
- Treat this as the bridge between reading the code and explaining your
	engineering choices in an interview or review setting.

---

## Implementation checklist (step-by-step)

1. Run `python3 tbd_problem.py` to generate `tbd_frames.bin`,
	 `tbd_params.bin`, and `tbd_reference.txt`.
2. Implement or review `tbd_solution.cu` ensuring exact binary parsing and
	 indexing semantics.
3. Compile with `nvcc` (adjust `-arch` to match GPU). Example:

```bash
nvcc -O3 -arch=sm_86 -o tbd_solution tbd_solution.cu
```

4. Run the solution and verify printed `(x0,y0,vx,vy,statistic)` matches
	 `tbd_reference.txt` exactly (or within acceptable float formatting).
5. Tune kernel config (`blockDim`), memory strategy (`__ldg` vs texture),
	 and reduction strategy to improve kernel_ms reported by CUDA events.

---

If you'd like, I can now:
- run a quick read to confirm this file was written correctly, and
- optionally update `tbd_solution.cu` or create a small test harness to
	automatically compare the CUDA output to `tbd_reference.txt`.

End of notes.

