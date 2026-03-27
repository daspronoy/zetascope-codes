Step 1 — Shared Memory Tiling: Fully Valid, Not Implemented
The highest-impact optimization. The roadmap's frame-chunking strategy (§3.1) directly addresses the L1/L2 uncoalesced reads. The 10–30× estimate is plausible. The tile sizing math (128×128 base + ~24px velocity footprint for C=8 chunk, 23 KB shared memory) is correct for Orin's 48 KB shared memory config.

Step 2 — Register Pressure: Partially Done
__launch_bounds__ is set, but the comment acknowledges "~35-40 registers/thread." Whether spilling to local memory is occurring requires ncu --set full. The roadmap's table is still accurate — at 40 regs with 256 threads/block you get 83% occupancy, not 100%.

Step 3 — Coalescing Audit: Valid for the tile load
Once tiling is implemented, the cooperative load itself must be coalesced. The sectors/requests ratio verification command (§3.3) is directly applicable.

Step 4 — L2 Persistence: Valid, Not Implemented
The cudaStreamAttributeAccessPolicyWindow code in §3.4 is not in v4. With Orin's 4 MB L2, this is a real opportunity especially for repeated tile loads.

Step 5 — FP16 Accumulation: Valid, Not Implemented
evaluate_trajectory accumulates in FP32. The inner loop sums ~300 terms — FP16 for partial sums within a chunk (8–16 terms) is safe and could give 2× throughput on the accumulation path.

Section 5.3 — Stream Pipelining: Valid, Not Implemented
The host-side INT8 conversion loop (~5.70s) and cudaMemcpy are sequential before any kernels launch. Pipelining the conversion with kernel execution is a free win.