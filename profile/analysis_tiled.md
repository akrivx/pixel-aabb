# 2x2 tile optimisation: profile comparison
## `find_aabbs_shared` vs `find_aabbs_tiled` -- RTX 2070 (sm_75, Turing)

---

## 1. Headline result

| Metric | find_aabbs_shared | find_aabbs_tiled | Change |
|---|---|---|---|
| Duration | 143 us | 43 us | **3.3x faster** |
| Warp Cycles Per Issued Instruction | 60.3 | 38.4 | 1.57x improvement |
| SM Busy | 11.81% | 15.99% | 1.35x improvement |
| Eligible Warps Per Scheduler | 0.22 | 0.32 | 1.45x improvement |
| Issue rate | 1 per 8.1 cycles | 1 per 5.3 cycles | 1.53x improvement |
| DRAM Throughput | 9.93% | 25.73% | 2.6x increase |
| L1/TEX Hit Rate | 0% | 49.22% | new (tile reuse) |
| L2 Cache Throughput | 3.10% | 9.10% | 2.9x increase |
| L1/TEX Cache Throughput | 86.29% | 74.48% | lower (less MIO) |
| Memory Throughput | 42.69 GB/s | 111.57 GB/s | 2.6x increase |
| Executed Instructions | 2,941,571 | 1,257,120 | 2.34x fewer |
| Grid (blocks) | 4096 | 1024 | 4x fewer |
| Waves Per SM | 28.44 | 7.11 | 4x fewer |
| Divergent Branches/Warp | 34.89 | 54.56 | increase (new) |
| Branch Efficiency | 97.63% | 91.84% | slight drop |

The kernel is **3.3x faster** and **8.7x faster than the original naive baseline**.

---

## 2. What the tiled kernel changed and why

### 2.1 The primary change: fewer instructions, not fewer atomics per se

The most striking number is **executed instructions: 1.26M vs 2.94M in shared**. That is a 2.34x reduction even though each thread now processes 4 pixels instead of 1. How?

**find_aabbs_shared** (4096 blocks × 256 threads each):
- Phase 1 init: 5 shared arrays × 256 entries × 4096 blocks = **5.24M shared-memory writes**
- Phase 2: one pixel per thread → 1 coalesced global load, then up to 5 shared-memory atomics per non-background thread
- Phase 3: a handful of global atomics per block

**find_aabbs_tiled** (1024 blocks × 256 threads each):
- Phase 1 init: 5 arrays × 256 entries × 1024 blocks = **1.31M shared-memory writes** (4x less)
- Phase 2: 4 global loads per thread, but for a spatially uniform tile all 4 pixels produce only **1 shared-memory commit** instead of 4. Each commit is 5 operations (4 atomics + 1 atomicExch) -- same as before, just called 4x less often.
- Phase 3: same

The phase 1 reduction alone saves 3.93M MIO instructions. Phase 2 reduces atomic commits roughly 4x for typical spatially coherent input. Total MIO pressure drops by approximately 4-5x, directly reducing the MIO Throttle stall that limited find_aabbs_shared.

### 2.2 MIO Throttle is no longer the dominant stall

In find_aabbs_shared, MIO Throttle was **44.9%** of warp stall cycles (27 cycles out of 60 per issued instruction). The MIO queue — which handles all shared-memory instructions — was perpetually backlogged.

In find_aabbs_tiled, Warp Cycles Per Issued Instruction dropped from **60.3 to 38.4**. The reduction is consistent with MIO Throttle being largely resolved. The L1/TEX Throughput fell from 86.3% to 74.5%, confirming that fewer shared-memory operations are flowing through the L1 pipeline.

The ncu details file notes: "Sampling interval is larger than 10% of the workload duration, which likely results in very few collected samples." The kernel runs only 43 us; the profiler sampling interval of 20,000 cycles at 1.17 GHz is ~17 us, meaning only ~2-3 samples per kernel run. The warp stall percentage breakdown is therefore unreliable for this kernel. We infer the bottleneck from the other metrics instead.

### 2.3 The new bottleneck: Long Scoreboard (global memory latency)

Evidence:
- **No Eligible: 81.17%** -- warps are stalled 81% of cycles, but the reason has shifted from MIO to memory
- **DRAM Throughput: 25.73%** vs 9.93% for shared -- nearly 3x more DRAM traffic relative to kernel duration; the seg read now dominates the memory workload
- **L1/TEX Hit Rate: 49.22%** -- explained in detail below; half the global loads serve from L1, half miss to L2/DRAM
- **L2 Throughput: 9.10%** vs 3.10% -- more L2 traffic, consistent with more DRAM misses
- **Memory Throughput: 111.57 GB/s** vs 42.69 GB/s -- 2.6x more total bandwidth consumed

The global memory load pattern is: each thread reads 4 pixels at (px, py), (px+1, py), (px, py+1), (px+1, py+1). Across the warp, thread `tx` reads pixels at x = 2*tx and 2*tx+1. These are stride-2, so a 32-thread warp touches 64 consecutive integers in a row -- two 128-byte cache lines -- rather than one. This is less coalesced than find_aabbs_shared (which had 32 consecutive threads reading 32 consecutive ints = perfect coalescing). The price is 2x more L2 transactions per row loaded. The benefit is that each row is only loaded once across 4 pixel accesses per thread, which explains the 49% L1 hit rate:

- **Pixel 1 (dx=0, dy=0)**: warp reads the even elements of two cache lines → 2 L2 misses
- **Pixel 2 (dx=1, dy=0)**: warp reads the odd elements of the SAME two cache lines → 2 L1 hits
- **Pixel 3 (dx=0, dy=1)**: warp reads the even elements of two new cache lines (next row) → 2 L2 misses
- **Pixel 4 (dx=1, dy=1)**: warp reads the odd elements of the same two cache lines → 2 L1 hits

4 accesses total: 2 miss, 2 hit → **50% hit rate**. The measured 49.22% matches perfectly.

The total DRAM data loaded remains the same 4 MB (every pixel read once), but the access pattern results in 2x more L2 transactions than find_aabbs_shared for the same data. The kernel is not DRAM-bandwidth bound (25.73% of 448 GB/s = ~115 GB/s; peak for this workload size would be higher). The 81% warp idle time is caused by warps stalling on L2/DRAM load latency (~200-400 cycles per miss) while waiting for the seg read.

### 2.4 Instruction count: 2.94M → 1.26M despite 4x more pixels per thread

The shared kernel executed 2.94M instructions for 1M pixels = 2.94 instructions per pixel.
The tiled kernel executes 1.26M instructions for 1M pixels = 1.26 instructions per pixel -- 2.3x more efficient.

Why fewer instructions per pixel?
- Phase 1 init: shared runs 5.24M shared writes across 4096 blocks. Tiled runs 1.31M. Saving: 3.93M instructions -- the biggest single factor.
- Phase 2: the run-length accumulation in registers is cheap (a compare + conditional branch + register min/max). The 4 global loads per thread now happen in a single phase instead of 4 separate kernel phases. The shared-memory commits (the expensive part) happen ~1x per tile instead of ~4x.
- Phase 3: 4x fewer blocks means 4x fewer global-atomic flush operations.

More instructions that are cheap (register ops, L1 loads) is better than fewer instructions that are expensive (shared-memory atomics, L2/DRAM loads). This is exactly the same principle as the move from find_aabbs_naive to find_aabbs_shared.

### 2.5 Branch divergence increased: 34.89 → 54.56 divergent branches per warp

find_aabbs_shared had divergence in phase 3 only (`if (tid < s_unique_count)`). find_aabbs_tiled adds divergence from the PROCESS_PIXEL macro: the conditional `if (id != 0 && id == cur_id)` takes different paths depending on whether each thread's tile has uniform IDs. In a warp where some tiles are uniform (all same ID) and others straddle two objects, threads will diverge on the mid-tile ID-change branch.

Branch efficiency dropped from 97.63% to 91.84%. This is a real but minor overhead. With 38.4 cycles per issued instruction and 81% of cycles stalled on memory, divergence is not the bottleneck -- those divergent branches add execution overhead only in the 19% of non-stalled cycles.

### 2.6 Waves Per SM: 28.44 → 7.11 (4x reduction)

A "wave" is one round of blocks scheduled across all SMs simultaneously. Each SM can hold 4 blocks (Block Limit Warps = 4, each block uses 8 warps, 4 × 8 = 32 warps = 100% of SM warp capacity).

- find_aabbs_shared: 4096 blocks / 36 SMs = 113.8 blocks per SM → 113.8 / 4 = **28.44 waves**
- find_aabbs_tiled: 1024 blocks / 36 SMs = 28.4 blocks per SM → 28.4 / 4 = **7.11 waves**

With 7 waves instead of 28, there are 4x fewer block-launch/drain transitions. The first and last wave are partially utilised (not all SMs receive a block), but 7 waves means this overhead is proportionally smaller. This contributes to the lower achieved occupancy (88.75% vs 93.56%) but also means less tail-latency at the end of the kernel.

---

## 3. Side-by-side comparison

| Section | Metric | Naive | Shared | Tiled | Notes |
|---|---|---|---|---|---|
| Speed of Light | Duration | 447 us | 143 us | 43 us | 3.3x / 8.7x speedup |
| Speed of Light | Compute Throughput | 9.50% | 13.99% | 15.99% | SM more active each step |
| Speed of Light | Memory Throughput | 12.44% | 43.14% | 37.24% | L1 now dominant |
| Speed of Light | DRAM Throughput | 3.24% | 9.93% | 25.73% | seg read now main traffic |
| Speed of Light | L1 Throughput | 20.42% | 86.29% | 74.48% | lower: fewer MIO ops |
| Speed of Light | L2 Throughput | 12.44% | 3.10% | 9.10% | strided access vs coalesced |
| Compute | SM Busy | 0.87% | 11.81% | 15.99% | progressive improvement |
| Compute | Executed Ipc Active | 0.03 | 0.49 | 0.73 | 24x vs baseline |
| Scheduler | No Eligible | 99.11% | 87.63% | 81.17% | stalls reducing each step |
| Scheduler | Eligible Warps | 0.01 | 0.22 | 0.32 | more schedulable warps |
| Scheduler | Issue rate | 1/113 cyc | 1/8.1 cyc | 1/5.3 cyc | 21x vs baseline |
| Warp State | Cycles Per Issue | 780 | 60.3 | 38.4 | primary bottleneck metric |
| Warp State | Primary stall | LG Throttle 72% | MIO Throttle 45% | Long Scoreboard (inferred) |
| Memory | L2 Hit Rate | 92.17% | 8.23% | 5.20% | streaming dominates |
| Memory | L1 Hit Rate | 0% | 0% | 49.22% | tile reuse (stride-2 pattern) |
| Memory | Memory Throughput | 14.09 GB/s | 42.69 GB/s | 111.57 GB/s | |
| Instructions | Executed | 665K | 2.94M | 1.26M | phase 1 + atomics drop |
| Launch | Grid | 4096 blocks | 4096 blocks | 1024 blocks | 4x fewer |
| Launch | Waves Per SM | 28.44 | 28.44 | 7.11 | 4x fewer |
| Launch | Registers/Thread | 16 | 32 | 32 | |
| Launch | Shared Mem/Block | 0 | 6.15 KB | 6.15 KB | |
| Occupancy | Achieved | 85.15% | 93.56% | 88.75% | |
| Branches | Divergent/Warp | 0 | 34.89 | 54.56 | tile ID logic adds divergence |
| Branches | Branch Efficiency | n/a | 97.63% | 91.84% | minor overhead |

---

## 4. Bottleneck progression

| Version | Dominant bottleneck | Kernel time | Speedup |
|---|---|---|---|
| Baseline (naive) | LG Throttle: global atomic serialisation, 71.6% stall | 447 us | 1x |
| Shared-memory reduction | MIO Throttle: shared atomic contention, 44.9% stall | 143 us | 3.1x |
| 2x2 tile (current) | Long Scoreboard: global load latency (inferred), 81% idle | 43 us | 10.4x |

The bottleneck has shifted from MIO Throttle (on-chip queue for shared-memory operations) to Long Scoreboard (waiting for global memory loads to return from L2/DRAM). Each progression has reduced one class of serialisation and exposed the next-level constraint.

---

## 5. Why the speedup is 3.3x and not higher

**Factor 1: Long Scoreboard stalls (81% idle warps).**
The kernel is still latency-limited. Warps stall waiting for global loads to return. With 4 loads per thread (vs 1 in find_aabbs_shared), the scheduler has more outstanding loads to hide but also more total latency exposure. The L1 hit rate of 49% helps (fast ~5-cycle returns for half the loads) but L2/DRAM misses for the other half take 200-400 cycles. To hide this, the scheduler needs many in-flight warps, but Block Limit Warps = 4 means each SM holds only 32 warps, leaving limited bandwidth for latency hiding.

**Factor 2: Stride-2 memory access pattern.**
Each warp loads 4 cache lines of segmentation data (2 per row, 2 rows) instead of the 1 cache line that find_aabbs_shared loads per pixel per warp. The total DRAM data is the same (4 MB), but L2 transactions double. This partially undermines the cache benefit of reading 4 pixels per thread.

**Factor 3: Increased branch divergence (54.56 divergent branches/warp).**
The per-pixel conditional logic in PROCESS_PIXEL causes more intra-warp divergence than find_aabbs_shared. Branch efficiency fell to 91.84%. While this is not the bottleneck, it consumes some fraction of the SM execution cycles that could otherwise issue productive instructions.

---

## 6. What would help next

### 6.1 Switch to uint8_t segmentation input

Currently the segmentation is stored as `int` (4 bytes per pixel). Switching to `uint8_t` (1 byte per pixel) would:
- Reduce DRAM traffic 4x: from 4 MB to 1 MB
- For find_aabbs_tiled, each warp loads stride-2 bytes instead of stride-2 ints. Two rows of 64 bytes (not ints) = 2 cache lines total vs 4 currently.
- L1 hit rate would rise further (more pixels per cache line)

At 25.73% DRAM utilisation, the kernel is approaching the regime where DRAM bandwidth starts to matter. The theoretical minimum to read 1 MB at 448 GB/s is ~2.2 us vs ~8.9 us for 4 MB. This alone could give a 2-4x improvement **once DRAM becomes the binding constraint**, which it will be sooner after the uint8_t switch.

The reason uint8_t showed no improvement for find_aabbs_shared was that DRAM was only 9.93% utilised -- the MIO bottleneck swamped any DRAM savings. With MIO now resolved, DRAM is 25.73% of the bottleneck and growing.

### 6.2 Larger tile (4x2 or 4x4)

A 4x2 tile (8 pixels per thread) would:
- Halve the grid to 512 blocks: phase 1 drops to 655K init writes
- For a uniform tile: still 1 shared-memory commit for 8 pixels (vs 1 for 4 currently)
- Keep stride-2 row reads but process 4 consecutive row pairs, maximising L1 reuse per thread
- Further hide L2/DRAM latency by keeping the thread busy with 8 loads and 8 comparisons

The risk is register pressure: more live variables per thread (tile coordinates, cur_id, lmnx/lmny/lmxx/lmxy across more iterations) may push registers above 32 per thread, triggering a Block Limit Registers reduction in blocks per SM.

### 6.3 Address the stride-2 access pattern

The current access is strided because thread tx reads pixels 2*tx and 2*tx+1 in x. An alternative layout: process tiles in column-major order within the block, or restructure the thread-to-tile mapping so that consecutive threads read consecutive pixels in both passes. This would require a more complex index calculation but could restore fully coalesced L2 access for both rows.

---

## 7. Updated performance trajectory

| Version | Dominant bottleneck | Kernel time | Cumulative speedup |
|---|---|---|---|
| Baseline (naive) | LG Throttle (global atomic serialisation) | ~447 us | 1x |
| + Shared-memory reduction | MIO Throttle (shared atomic contention) | ~143 us | 3.1x |
| + 2x2 tile (current) | Long Scoreboard (global load latency) | ~43 us | 10.4x |
| + uint8_t input | Long Scoreboard / DRAM bandwidth | ~15-25 us | est. 25-30x |
| + Larger tile (4x2) | DRAM bandwidth | ~10-15 us | est. 30-45x |
