changes"

1. Memory access coalescing in "evaluate_trajectory"
frames[t * frame_stride + py * nx + px] — each warp lane follows a different (vx, vy) trajectory, so px/py diverge across lanes. This means uncoalesced 128-byte cache lines are loaded from global memory for almost every frame access. On Orin's ~200 GB/s unified bus, this is the dominant bottleneck and neither strategy addresses it.

2. "reduce_argmax" kernel uses shared memory reduction
This is correct but ignores that a simple warp-shuffle reduction (as used in Strategy 4) would be faster and more consistent.

