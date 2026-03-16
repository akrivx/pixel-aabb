# Shared-memory reduction: profile comparison
## `find_aabbs_naive` vs `find_aabbs_shared` -- RTX 2070 (sm_75, Turing)

---

## 1. Headline result

| Metric | find_aabbs_naive | find_aabbs_shared | Change |
|---|---|---|---|
| Duration | 447.36 us | 141.95 us | **3.15x faster** |
| Warp Cycles Per Issued Instruction | 768 | 61 | 12.6x improvement |
| SM Busy | 0.87% | 11.68% | 13.4x improvement |
| Eligible Warps Per Scheduler | 0.01 | 0.22 | 22x improvement |
| Issue rate | 1 per 113 cycles | 1 per 8.1 cycles | 14x improvement |
| L2 Slice Imbalance (max vs avg) | +44% | +7.8% | 5.6x improvement |
| L1/TEX Cache Throughput | 20.4% | 85.4% | |
| L2 Cache Throughput | 12.5% | 3.0% | |
| Executed Instructions | 665,489 | 2,941,594 | 4.4x more |
| Registers Per Thread | 16 | 32 | doubled |
| Static Shared Memory Per Block | 0 | 6.15 KB | |

The kernel is 3.15x faster. Every scheduling and contention metric improved dramatically.
The bottleneck has shifted from a different stall type, described in detail below.

---

## 2. What changed and why

### 2.1 The primary bottleneck is gone

In the naive kernel, **LG Throttle** dominated at 71-72% of warp stall cycles. Every
non-background pixel issued four global atomics, filling the LSU's global memory instruction
queue faster than L2 could drain it.

In the shared kernel, global atomics in phase 3 are reduced to `s_unique_count * 4` per block
-- typically 4-16 in total, versus up to 1024 in the naive version. The LG queue is no longer
under meaningful pressure.

The evidence: **L2 Cache Throughput fell from 12.5% to 3.0%**. The naive kernel's L2 was
kept busy by constant atomic traffic on the AABB arrays. The shared kernel barely touches L2
for writes -- phase 3 is a handful of operations per block. Global memory traffic is now
dominated entirely by the segmentation read, which is a one-shot streaming access.

**L2 Slice Imbalance dropped from +44% to +7.8%.** In the naive kernel, popular objects'
AABB entries concentrated traffic on specific L2 slices. With almost no global atomics
remaining, L2 slices are now load-balanced -- each handles roughly an equal share of the
segmentation read traffic.

**L2 Hit Rate fell from 91.5% to 8.5%.** This looks like a regression but is not. In the
naive kernel, the AABB arrays (16 KB) were hammered so frequently they stayed hot in L2,
giving a high hit rate on repeated atomic reads. In the shared kernel, global memory traffic
is almost entirely the segmentation read (4 MB streamed once), which has no reuse and mostly
misses L2. The hit rate drop reflects the shift from repeated atomic reads to a streaming
load -- a good thing.

### 2.2 The new bottleneck: MIO Throttle

The shared kernel's dominant stall is now **MIO Throttle at 44.9%** (27.2 cycles out of
60.7 cycles per issued instruction).

MIO is the Memory Input/Output queue -- distinct from the LG queue -- which handles shared
memory instructions, special math operations, and dynamic branches. Phase 2 of the shared
kernel issues four shared-memory atomics per non-background pixel (`atomicMin`/`atomicMax`
into `s_min_x`, `s_min_y`, `s_max_x`, `s_max_y`), plus `atomicExch` for deduplication.
With 256 threads per block all hitting at most ~4 unique IDs in shared memory, the contention
within the block is high and the MIO queue fills up.

The MIO Throttle stall is structurally similar to the LG Throttle stall but cheaper. Shared
memory latency is ~5 cycles versus ~30-60 cycles for L2. The contention involves at most 256
threads per block rather than 1 million threads globally. This is why the kernel is 3x faster
despite still stalling heavily -- the stall cycles are shorter and less severe.

### 2.3 L1/TEX Throughput: 20% -> 85%

On Turing, shared memory is physically part of the L1/TEX subsystem. The 85.4% L1 throughput
reflects the heavy use of shared-memory atomics in phase 2. The MIO queue feeds the shared
memory banks, and the L1/TEX pipeline is now the most-utilised resource in the kernel. This
is consistent with MIO Throttle being the dominant stall: the L1 pipeline is under load but
the queue is the limiting factor, not the pipeline capacity itself.

### 2.4 Executed instructions: 665K -> 2.94M (4.4x more)

The shared kernel executes 4.4x more instructions while being 3.15x faster. This is correct
and expected:

- **Phase 1** initialises five 256-entry shared arrays (s_min_x, s_min_y, s_max_x, s_max_y,
  s_seen) across 4096 blocks. That is 5 * 256 * 4096 = 5.24M shared memory writes -- a
  significant instruction count just for setup.
- **Phase 2** issues the same segmentation read as before, plus shared memory atomics plus
  the atomicExch/atomicAdd for deduplication. Each pixel now generates more instructions than
  in the naive kernel.
- Those instructions are cheap (on-chip, low-latency) compared to the serialised global
  atomics in the naive kernel. More instructions at low latency is better than fewer
  instructions at high latency.

### 2.5 Registers per thread: 16 -> 32

The shared kernel uses twice as many registers. The compiler needs registers for: the shared
memory pointers, tid, block_size, x, y, id, slot, the loop induction variable, and the
various temporaries introduced by the three-phase structure.

This triggers Block Limit Registers = 8 blocks per SM (vs 16 for naive). However, Block Limit
Warps = 4 blocks per SM remains the binding constraint (each block uses 8 warps; 4 * 8 = 32
warps = 100% of SM warp capacity). Doubling register use therefore has zero occupancy impact
in this case.

### 2.6 Shared memory: 0 -> 6.15 KB per block

This is what the six shared arrays cost: 6 * 256 * 4 bytes = 6144 bytes = 6 KB per block.

Block Limit Shared Mem = 5 blocks per SM (32 KB total / 6 KB per block = 5 with rounding).
Again, Block Limit Warps = 4 is still the binding constraint, so the 6 KB allocation has no
occupancy penalty.

Achieved occupancy improved slightly from 85.5% to 93.7% -- the shared kernel keeps warps
more uniformly active across blocks, reducing the scheduling imbalance that was causing the
gap in the naive kernel.

### 2.7 Branch divergence appeared: 34.81 divergent branches per warp

In the naive kernel, branch divergence was zero. The shared kernel introduces real divergence:

- Phase 3: `if (tid < s_unique_count)` -- most threads skip the global atomics. Within a
  warp of 32 threads (tid 0..31), some have `tid < s_unique_count` (true) and others do not
  (false). The warp must execute both paths serially.
- The `if (id != 0)` background check and the `if (atomicExch(...) == 0)` dedup check also
  contribute.

Branch Efficiency is 97.6%, meaning 97.6% of branch instructions do not cause divergence.
The 34.81 divergent branches per warp is a real but minor cost. It is not the bottleneck.

---

## 3. Side-by-side comparison

| Section | Metric | Naive | Shared | Notes |
|---|---|---|---|---|
| Speed Of Light | Duration | 447 us | 142 us | 3.15x speedup |
| Speed Of Light | Compute Throughput | 9.46% | 13.84% | SM more active |
| Speed Of Light | Memory Throughput | 12.5% | 42.67% | L1 now dominant |
| Speed Of Light | DRAM Throughput | 3.52% | 8.57% | seg read now visible |
| Speed Of Light | L1 Throughput | 20.3% | 85.4% | shared mem traffic |
| Speed Of Light | L2 Throughput | 12.5% | 3.0% | atomics gone from L2 |
| Compute | SM Busy | 0.87% | 11.68% | 13.4x improvement |
| Compute | Executed IPC Active | 0.04 | 0.49 | 12x improvement |
| Scheduler | No Eligible | 99.12% | 87.58% | still high but improving |
| Scheduler | Eligible Warps | 0.01 | 0.22 | 22x improvement |
| Scheduler | Issue rate | 1/113 cycles | 1/8.1 cycles | 14x improvement |
| Warp State | Cycles Per Issue | 768 | 61 | 12.6x improvement |
| Warp State | Primary stall | LG Throttle 72% | MIO Throttle 45% | bottleneck shifted |
| Memory | L2 Hit Rate | 91.5% | 8.5% | streaming now dominates |
| Memory | L1 Hit Rate | 0% | 0% | seg read still streaming |
| Memory | Memory Throughput | 15.3 GB/s | 37.2 GB/s | |
| Instructions | Executed | 665K | 2.94M | 4.4x more, but cheap |
| Launch | Registers/Thread | 16 | 32 | doubled |
| Launch | Shared Mem/Block | 0 | 6.15 KB | |
| Occupancy | Achieved | 85.5% | 93.7% | |
| L2 Distribution | Max slice vs avg | +44% | +7.8% | contention eliminated |
| Branches | Divergent Branches | 0 | 34.8 | new but minor |

---

## 4. Why the speedup is 3.15x and not higher

The predicted range was 5-15x. The actual result is 3.15x. Two factors limit it:

**Factor 1: Phase 1 initialisation overhead.**
Before any useful work, every block must zero out five 256-entry shared arrays. That is 1280
shared memory writes per block, across 4096 blocks = 5.24M writes, all going through the MIO
queue. This cost does not exist in the naive kernel. In a production kernel processing many
frames, this is unavoidable per-frame overhead.

**Factor 2: MIO Throttle from shared memory atomics (44.9% of stall cycles).**
Phase 2 still issues four atomics per non-background pixel -- the same count as the naive
kernel, just targeting shared memory instead of global. Within each block, 256 threads race
on ~4 shared memory addresses. Shared memory atomics serialise per bank (not per address, but
per 32-byte bank -- and four addresses in a 256-int array may map to the same banks). The MIO
queue fills up similarly to how the LG queue filled up before, just at lower latency and
smaller scale.

---

## 5. The new bottleneck in detail: shared memory atomics

The MIO Throttle is the direct successor to the LG Throttle. The fix is the same in
principle: reduce the number of atomic operations before they reach shared memory, just as the
shared-memory reduction reduced global atomics before they reached L2.

The next step is a **warp-level reduction**: instead of all 32 threads in a warp independently
calling `atomicMin(&s_min_x[id], x)`, first reduce x within the warp using register shuffles
(`__shfl_down_sync`), then have only thread 0 of each warp write the warp's local minimum to
shared memory. This cuts phase 2 shared memory atomic traffic by 32x (from 256 per block to
8 per block for an 8-warp block).

On sm_80+ (Ampere), `__reduce_min_sync` / `__reduce_max_sync` do this in a single
instruction. On sm_75 (this GPU), the equivalent must be written manually with
`__shfl_down_sync`, but the result is the same.

However, warp-level reduction is complicated by the fact that threads within a warp may have
different `id` values. A warp covering a boundary between two objects cannot naively reduce
min_x across all threads -- it must first group threads by ID. This makes the reduction
considerably more complex than a simple min/max across a uniform warp.

A simpler intermediate step: **reduce contention by partitioning phase 2**. Rather than all
256 threads atomically updating the same 4 shared addresses per ID, assign each warp to
accumulate its own register-level min/max, then have one thread per warp commit to shared
memory. This is the manual shuffle approach and removes the MIO Throttle without requiring
hardware warp reduction intrinsics.

---

## 6. Updated performance trajectory

| Version | Dominant bottleneck | Kernel time | Notes |
|---|---|---|---|
| Baseline (naive) | LG Throttle (global atomic serialisation) | ~447 us | 99% idle cycles |
| + Shared-memory reduction (current) | MIO Throttle (shared atomic contention) | ~142 us | 3.15x speedup achieved |
| + Warp-level reduction (manual shuffle) | DRAM bandwidth or init overhead | ~30-60 us | Estimated 2-5x further gain |
| + uint8_t input | DRAM bandwidth | ~15-30 us | 4x less DRAM traffic |

The uint8_t change is now more relevant: with the MIO stall reduced by warp shuffles, the
kernel will approach being genuinely DRAM-bandwidth bound on the segmentation read.
At that point, reading 1 MB instead of 4 MB will have direct impact on kernel time.

---

## 7. Profiling checkpoints for the next optimisation

After implementing warp-level reduction in phase 2:

- **MIO Throttle** should drop from 44.9% toward 5-10%.
- **Warp Cycles Per Issued Instruction** should drop further from 61 toward 10-20.
- **L1/TEX Throughput** should drop (fewer shared memory ops per cycle).
- **SM Busy** should rise above 11.68%.
- **Kernel duration** should drop by 2-5x toward 30-60 us.

If instead the kernel becomes **DRAM-bound** (DRAM Throughput rising toward 50-80%),
the next action is switching the segmentation buffer to uint8_t.
