# Baseline kernel performance analysis
## `find_aabbs_naive` -- RTX 2070 (sm_75, Turing)

---

## 1. Environment and run configuration

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2070 |
| Compute capability | sm_75 (Turing) |
| SM count | 36 |
| Warp schedulers per SM | 4 (each SM is divided into 4 sub-partitions, SMSPs) |
| Max warps per SM | 32 (8 per scheduler) |
| L1 cache per SM | 64 KB (shared with shared memory) |
| L2 cache (total) | 4 MB across 4 slices |
| Peak DRAM bandwidth | ~448 GB/s (GDDR6, 256-bit bus) |
| SM clock (measured) | 1.19 GHz |
| Grid | 64 x 64 = 4,096 blocks |
| Block | 16 x 16 = 256 threads (8 warps) |
| Total threads | 1,048,576 |
| Registers per thread | 16 |
| Shared memory per block | 0 bytes |
| Kernel duration | 450.59 us |
| Elapsed cycles | 540,706 |

---

## 2. Architecture primer: how Turing executes a kernel

Understanding the findings requires a clear mental model of what happens between a thread issuing an instruction and that instruction completing.

### 2.1 SMs, warps, and schedulers

The GPU dispatches blocks to SMs. Each SM runs multiple blocks concurrently (subject to resource limits -- registers, shared memory, warp slots). Within a block, 32 threads execute together as a **warp**. All threads in a warp execute the same instruction in lockstep; if some threads follow a different branch, the warp executes both paths serially with the inactive threads predicated off (warp divergence).

Each Turing SM has 4 warp schedulers (SMSPs). Each scheduler owns a pool of up to 8 warps and issues one instruction per cycle from a ready warp. An instruction is "ready" (eligible) when all its source operands are available and the relevant execution unit is free.

### 2.2 Latency hiding through warp switching

The fundamental strategy for hiding memory latency on a GPU is **latency hiding via warp switching**: when a warp issues a load and must wait tens or hundreds of cycles for the data, the scheduler immediately switches to another warp and issues its instruction. As long as there are enough active warps, the scheduler always has something productive to do and the latency is hidden.

For this to work, there must be a supply of eligible warps. If all active warps are stalled -- waiting for the same bottleneck -- no latency hiding is possible and the SM stalls.

### 2.3 The Load/Store Unit (LSU) and the LG queue

Global and local memory instructions (loads, stores, atomics) pass through the LSU. Inside the LSU is the **LG instruction queue**, which is a finite-depth buffer holding in-flight global memory operations. If the queue is full, the scheduler cannot issue another global memory instruction to that LSU partition. Any warp waiting to issue a global memory instruction must stall until a slot opens. This stall reason is called **LG Throttle**.

### 2.4 Atomic operations on Turing

On Turing (sm_75), `atomicMin` and `atomicMax` on global `int` arrays are **L2 atomics**: they are forwarded to the L2 cache, where the read-modify-write is performed in hardware. Each atomic operation takes a full round trip to L2 (~30-60 cycles). More importantly, atomics to the **same cache line must serialise**: the L2 can only process one atomic per cache line at a time. If 100 threads all call `atomicMin(&min_x[id], x)` for the same `id`, those 100 operations execute one after another. The second thread's atomic cannot begin until the first has completed.

This is the central architectural mechanism driving the results below.

---

## 3. Metric-by-metric analysis

### 3.1 GPU Speed Of Light (SOL) -- overall throughput utilisation

| Metric | Value | Notes |
|---|---|---|
| Duration | 450.59 us | Wall-clock kernel time |
| Compute (SM) Throughput | 9.45% | Fraction of peak instruction throughput used |
| Memory Throughput | 13.67% | Fraction of peak memory subsystem throughput used |
| DRAM Throughput | 8.24% | Fraction of peak DRAM bandwidth used |
| L1/TEX Cache Throughput | 20.31% | |
| L2 Cache Throughput | 13.67% | |
| SM Active Cycles | 533,573 | Cycles where at least one warp was active |

SOL shows that neither compute nor memory is being meaningfully utilised. Both figures are far below 60%, which ncu flags as an indicator of a latency-bound workload rather than a resource-saturated one. The GPU is not running out of compute capacity or memory bandwidth; it is spending almost all of its time waiting.

A kernel that is legitimately compute-bound would show Compute Throughput near 100%. A memory-bandwidth-bound kernel would show Memory Throughput near 100%. Neither is true here. The bottleneck is latency -- specifically, atomic serialisation latency blocking the instruction pipeline.

### 3.2 Compute Workload Analysis

| Metric | Value |
|---|---|
| Executed IPC Active | 0.03 inst/cycle |
| Issue Slots Busy | 0.87% |
| SM Busy | 0.87% |

**IPC (Instructions Per Cycle)** measures how many instructions a scheduler issues per cycle when active. A single-issue in-order scheduler has a theoretical maximum of 1.0. An IPC of 0.03 means the scheduler is issuing one instruction per 33 active cycles on average.

**Issue Slots Busy** at 0.87% means that out of every 100 cycles, the scheduler has a ready instruction to issue in less than 1 of them. The remaining 99+ cycles are wasted. This is one of the clearest possible expressions of a stall-dominated kernel.

### 3.3 Scheduler Statistics -- the stall picture

| Metric | Value |
|---|---|
| One or More Eligible | 0.89% |
| No Eligible | 99.11% |
| Active Warps Per Scheduler | 6.89 |
| Eligible Warps Per Scheduler | 0.01 |
| Effective issue rate | 1 instruction per 112.5 cycles |

There are, on average, 6.89 active warps per scheduler -- a reasonable number (maximum is 8). Active warps are warps that are resident on the SM and could in principle execute. However, **eligible** warps are those where every source operand is ready and no structural hazard blocks issue. Only 0.01 warps per scheduler are eligible at any given cycle.

The gap between active (6.89) and eligible (0.01) is the stall: the warps are present, but they are all waiting for something. The 99.11% "No Eligible" figure confirms that in 99 out of every 100 cycles, every single active warp is blocked.

### 3.4 Warp State Statistics -- where the time goes

| Metric | Value |
|---|---|
| Warp Cycles Per Issued Instruction | 775.24 |
| Warp Cycles Per Executed Instruction | 780.72 |
| Avg. Active Threads Per Warp | 31.39 |
| LG Throttle stall contribution | 555.3 cycles (71.6%) |

Each instruction requires 775 warp-cycles to be issued. This is the average number of cycles a warp spends between one instruction completing and the next being issued. A fully pipelined kernel with no stalls would achieve values of 4-8 here. At 775, the pipeline is almost completely stalled.

Of those 775 stall cycles, **555 are attributed to LG Throttle** -- the LG instruction queue is full. The kernel is issuing global memory operations (the four atomics per pixel) faster than the LSU can drain them. New memory instructions cannot be issued until old ones complete. Because the atomics serialise at L2 and each takes tens of cycles, the queue fills immediately and stays full.

The remaining ~220 stall cycles include other categories (scoreboard waits, memory dependencies) that are secondary.

**Average active threads per warp: 31.39** (maximum 32). Thread utilisation is high -- the warp divergence from background pixels (`if (id == 0) return`) is small, meaning most threads in every warp are doing real work. This rules out divergence as a concern.

### 3.5 Memory Workload Analysis

| Metric | Value |
|---|---|
| Memory Throughput | 35.86 GB/s |
| Mem Busy | 13.67% |
| L1/TEX Hit Rate | 0% |
| L2 Hit Rate | 82.99% |
| Mem Pipes Busy | 9.45% |

**L1 hit rate: 0%.** The segmentation buffer is 1024 x 1024 x 4 bytes = 4 MB. Each pixel is read exactly once by exactly one thread; there is no temporal or spatial reuse across threads from different blocks. With no reuse, every load misses L1 and goes to L2. This is expected behaviour for a streaming read pattern, not a bug.

**L2 hit rate: 82.99%.** The four AABB arrays (min_x, min_y, max_x, max_y) are each 256 ints = 1 KB. All four together are 4 KB. The RTX 2070 has 4 MB of L2, so the AABB arrays fit hundreds of times over. Once they are loaded from DRAM on the first access, they stay in L2 for the entire kernel. The high L2 hit rate reflects this: every atomic read-modify-write hits L2 without going to DRAM. This is why DRAM throughput (8.24%) is low despite significant memory activity.

**Effective DRAM throughput: 35.86 GB/s** against a peak of 448 GB/s (8%). This is almost entirely the cost of reading the 4 MB segmentation buffer as `int`. The AABB arrays contribute negligibly. The DRAM bandwidth is not saturated and is not the binding constraint.

### 3.6 Occupancy

| Metric | Value |
|---|---|
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 84.56% |
| Block Limit (warps) | 4 blocks |
| Block Limit (registers) | 16 blocks |
| Block Limit (shared memory) | 16 blocks |
| Theoretical Active Warps per SM | 32 |
| Achieved Active Warps per SM | 27.06 |

**Theoretical occupancy is 100%.** The binding resource constraint is the warp limit: each block uses 8 warps (256 threads / 32 threads per warp), and a Turing SM supports 32 warps, so 4 blocks fit per SM without hitting any other limit. 4 blocks x 8 warps = 32 warps = 100% of the warp slots.

**Achieved occupancy is 84.6%** (27 warps instead of 32). The gap is minor and ncu estimates only a 15% speedup from closing it. More importantly, occupancy is **not the root problem**: even with 27 active warps per scheduler (6.89 per sub-partition), almost none of them are eligible. Adding more warps would not help because they would all stall on the same atomic serialisation bottleneck.

### 3.7 GPU and Memory Workload Distribution -- L2 slice imbalance

| Metric | Value |
|---|---|
| Max L2 slice active cycles vs. average | +44.96% above average |
| Min L2 slice active cycles vs. average | -15.53% below average |

The RTX 2070 has 4 L2 slices (one per 64-bit segment of the 256-bit memory bus). Cache lines are distributed across slices by a hash of their address. Ideally all slices are equally busy.

The +44.96% imbalance means one L2 slice is handling nearly 45% more traffic than the average slice. This is caused by the AABB arrays: multiple objects' `min_x`, `min_y`, `max_x`, `max_y` entries happen to hash to the same L2 slice. All atomics for those objects funnel through that slice, serialising against each other at the slice level even beyond the per-address serialisation described above.

This secondary effect compounds the contention: even if two threads are updating different object IDs, if those IDs' arrays hash to the same L2 slice, they may still queue behind each other at the slice.

### 3.8 Coalescing and instruction efficiency

| Metric | Value |
|---|---|
| Branch Instructions Ratio | 0.12% |
| Avg. Divergent Branches | 0 |
| Excessive L2 sectors (uncoalesced) | 8 (0.004% of 194,580 total) |

Memory access coalescing is essentially perfect. A warp of 32 threads reads 32 consecutive `int` pixels (the segmentation row stride is the image width, and `x` increments by 1 per thread), which maps to exactly one 128-byte cache line per 32 threads. The hardware combines those 32 loads into a single L2 request -- this is coalescing working correctly.

There are only 8 "excessive" (uncoalesced) sectors in the entire kernel execution. Coalescing is not a concern.

Branch divergence is negligible (0 divergent branches). The predicated-off threads for background pixels (`if (id == 0) return`) are handled efficiently.

---

## 4. Root cause summary

The kernel is overwhelmingly bottlenecked by **atomic serialisation at the L2 cache**, manifesting as **LG Throttle stalls (71.6% of all stall cycles)**.

The causal chain is:

1. Every non-background pixel issues four global atomic operations (`atomicMin`/`atomicMax` into `min_x`, `min_y`, `max_x`, `max_y`).
2. All threads processing pixels belonging to the same object target the same four memory addresses.
3. At the L2 cache, atomic operations to the same cache line must serialise. With thousands of threads per object, the effective serialisation queue is very long.
4. Each SM's LSU has a finite-depth LG instruction queue. The queue fills immediately because atomics take many cycles to complete, and new ones are arriving faster than old ones drain.
5. When the LG queue is full, the warp scheduler cannot issue further global memory instructions. All active warps stall.
6. With all warps stalled, no latency hiding is possible. The SM issues almost no instructions (0.87% of cycles), achieving 9.45% compute throughput and 13.67% memory throughput despite running 1 million threads.

Secondary observations:
- The `int` storage for pixel IDs wastes 4x DRAM bandwidth on the segmentation read (4 MB instead of the 1 MB needed for `uint8_t`).
- L2 slice imbalance (+44.96%) adds further contention above what per-address serialisation alone would cause.
- Occupancy and coalescing are both in good shape and are not contributing problems.

---

## 5. Recommended optimisations

### Optimisation 1: per-block shared-memory reduction

**What:** Each thread block computes a local AABB in shared memory first. After all threads in the block have contributed, a single elected thread issues one global atomic per object ID found in the block.

**Why the data supports this:**
The LG Throttle stall accounts for 71.6% of warp execution time. It is caused by global atomic volume, not by the cost of any individual atomic. Reducing the number of global atomics is the direct fix.

Current global atomic count: roughly (number of non-background pixels) x 4. For a 1024x1024 image with 20 objects, approximately 800,000 non-background pixels x 4 = 3.2 million global atomics.

With shared-memory reduction: each block produces at most one global atomic per object ID per dimension. A block covering 256 pixels touches at most a handful of distinct object IDs. For a representative object spanning 180x120 pixels, it intersects approximately ceil(180/16) x ceil(120/16) = 12 x 8 = 96 blocks. Those 96 blocks issue 96 global atomics for that object instead of 21,600. The reduction factor for that object alone is 225x.

Across the whole image: roughly 4,096 blocks, each issuing perhaps 4-8 global atomics (for the 1-2 objects it contains), giving ~30,000 global atomics instead of 3.2 million. That is a roughly 100x reduction in atomic traffic.

**Expected outcome:** The LG Throttle stall should collapse from 71.6% to a small fraction. The kernel will transition from atomic-latency-bound to memory-bandwidth-bound on the segmentation read. Expected speedup: 5-15x, bringing kernel time from ~450 us down to roughly 30-90 us, depending on how effectively the shared-memory reduction eliminates the queue pressure.

**Shared memory cost:** For a 16x16 block processing up to N distinct object IDs, the shared memory needed is 4 x N x sizeof(int). If only one pass through the block's pixels is needed and the number of unique IDs per block is small (1-4 in practice), the cost is 16-64 bytes -- negligible compared to the 32+ KB available per SM. Even storing a full 256-entry local AABB table (4 x 256 x 4 = 4 KB) leaves ample shared memory headroom.

**Impact on occupancy:** Adding shared memory per block reduces the number of blocks that fit per SM (Block Limit Shared Mem is currently unconstrained at 16 blocks). Even a 4 KB shared memory allocation reduces the shared-memory block limit from 16 to 8 (32 KB total / 4 KB per block). Since the warp limit already constrains to 4 blocks, a 4 KB allocation has zero occupancy impact. Only if the allocation exceeds 8 KB would occupancy begin to drop.

---

### Optimisation 2: uint8_t segmentation storage

**What:** Change the segmentation buffer from `int` (4 bytes per pixel) to `uint8_t` (1 byte per pixel), both on the host and on the GPU. Adjust the kernel to read `uint8_t` and convert to `int` for the AABB arrays.

**Why the data supports this:**
The segmentation data is 1024 x 1024 x 4 bytes = 4 MB read from DRAM. Object IDs are in the range 0-255 and fit in a single byte. The `int` storage wastes 3 bytes per pixel -- 75% of the DRAM bandwidth spent on padding.

With `uint8_t`: 1024 x 1024 x 1 byte = 1 MB. A 128-byte cache line holds 128 pixels instead of 32, quadrupling the useful data delivered per L2/DRAM transaction. DRAM read traffic drops from 4 MB to 1 MB.

Current DRAM throughput is 35.86 GB/s against a 448 GB/s peak (8%). After the shared-memory fix, the kernel will be bandwidth-bound on the segmentation read. With `int`, reading 4 MB at 35 GB/s takes ~114 us. With `uint8_t`, reading 1 MB takes ~28 us. The combined effect of both optimisations should bring kernel time below 50 us.

**Coalescing with uint8_t:** A warp of 32 threads reading consecutive `uint8_t` pixels issues a 32-byte load, which fits in a single 128-byte cache line. Coalescing is maintained. The current coalescing efficiency (0.004% excessive sectors) is already perfect and will remain so.

**Expected outcome:** ~4x reduction in DRAM read traffic for the segmentation buffer. By itself -- without fixing the atomics -- this has minimal impact because the bottleneck is atomic latency, not DRAM bandwidth. Applied after the shared-memory fix, when the kernel is bandwidth-bound, this produces a further 2-4x speedup.

---

### Optimisation 3: warp-level reduction (sm_80+ only)

**What:** Use CUDA warp-level primitives (`__reduce_min_sync`, `__reduce_max_sync`, available from sm_80 / Ampere) to reduce within a warp before writing to shared memory. This eliminates shared memory bank conflicts and reduces the number of shared-memory writes per warp from 32 to 1.

**Applicability:** The RTX 2070 is sm_75 (Turing). These intrinsics require sm_80 (Ampere). This optimisation cannot be applied to this GPU. It is listed for completeness and for when the code is run on an Ampere or later device.

---

### Optimisation 4: L2 slice load balancing

**What:** Interleave the entries in the AABB arrays so that different object IDs map to different L2 slices. This can be achieved by padding each array to a cache-line boundary and arranging storage so that consecutive IDs land in different slices.

**Why the data supports this:**
The +44.96% L2 slice imbalance shows that atomic traffic is concentrated on a subset of slices. Even after the shared-memory fix reduces total atomic volume, slice imbalance means the remaining atomics still cluster. Distributing them evenly across slices would improve parallel L2 throughput.

**Expected outcome:** Secondary gain. Likely 10-20% improvement in atomic throughput after the primary fixes are in place, by eliminating the per-slice serialisation bottleneck. Not worth measuring until optimisations 1 and 2 are applied.

---

## 6. Expected performance trajectory

| Version | Dominant bottleneck | Estimated kernel time | Notes |
|---|---|---|---|
| Baseline (current) | LG Throttle (global atomic serialisation) | 450 us | 99% idle cycles |
| + Shared-memory reduction | DRAM bandwidth (segmentation read, int) | ~50-100 us | LG stall eliminated; reading 4 MB at ~35-50 GB/s |
| + uint8_t input | DRAM bandwidth (segmentation read, uint8_t) | ~15-30 us | 4x less DRAM traffic; approaching bandwidth limit |
| + L2 load balancing | Occupancy / instruction overhead | ~10-20 us | Diminishing returns; L2 contention distributed |

These estimates are order-of-magnitude. Actual results depend on object distribution in the image (fewer, larger objects mean fewer global atomics after the shared-memory fix), cache pressure from concurrent workloads, and achievable DRAM bandwidth at the working set size.

The most important validation step is to re-run ncu after each change and confirm that the dominant stall reason has shifted as predicted before proceeding to the next optimisation.

---

## 7. Recommended profiling checkpoints

After implementing optimisation 1 (shared-memory reduction):
- **Warp Cycles Per Issued Instruction** should drop from 775 to below 50.
- **LG Throttle stall** should drop from 71.6% to under 5%.
- **DRAM Throughput** should rise as the kernel becomes bandwidth-bound.
- **Kernel duration** should drop by at least 5x.

After implementing optimisation 2 (uint8_t):
- **DRAM Throughput %** should rise (same data volume covered faster, or same time with 4x less data).
- **dram_read_transactions** count should drop to approximately 1/4 of the post-optimisation-1 value.
- **Kernel duration** should drop by a further 2-4x.

If after both optimisations the kernel is still significantly below peak DRAM bandwidth, examine the **Scheduler Statistics** again -- a new bottleneck (register pressure, shared memory bank conflicts, occupancy loss) may have become visible.
