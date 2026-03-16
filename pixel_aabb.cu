// pixel_aabb.cu
//
// GPU pixel-AABB finder -- three kernel implementations:
//
//   find_aabbs_naive   -- baseline: one thread per pixel, four global atomics
//                         per non-background pixel.  Heavy L2 atomic contention.
//
//   find_aabbs_shared  -- optimisation 1: per-block shared-memory reduction.
//                         Threads reduce into shared memory first, then one
//                         thread per unique object ID in the block issues a
//                         single set of four global atomics.
//
//   find_aabbs_tiled   -- optimisation 2: each thread processes a 2x2 tile.
//                         Bounds are accumulated in registers across consecutive
//                         same-ID pixels; a shared-memory atomic is only issued
//                         when the ID changes or the tile ends.  For spatially
//                         coherent objects (the common case), each thread issues
//                         one commit instead of four, cutting MIO queue pressure.
//                         The 4x fewer blocks also reduce phase-1 init overhead.
//
// Loads a grayscale segmentation PNG (pixel value = object ID, 0 = background).
//
// Usage:  pixel_aabb [path/to/segmentation.png]
//         Defaults to assets/segmentation.png relative to the source tree (absolute
//         path baked in at build time via DEFAULT_SEG_PATH compile definition).
//
// Outputs:
//   result.png  -- colourised segmentation with white bounding-box overlays

#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "vis.h"

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <cstdint>

// One slot per possible 8-bit object ID (grayscale PNG -> IDs 0..255).
// Slot 0 = background (skipped by the kernel). Memory cost is negligible.
static constexpr int MAX_IDS = 256;

// --- CUDA error-checking macro -----------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// --- Kernel 1: naive baseline ------------------------------------------------
//
// One thread per pixel. Each non-background thread issues four atomics directly
// into global memory. All threads for the same object contend on the same four
// addresses in L2, serialising the atomic queue and leaving SMs ~99% idle.
//
// Profiler findings (RTX 2070, 1024x1024 image, 20 objects):
//   - Kernel time:            ~450 us
//   - LG Throttle stall:       71.6% of warp cycles
//   - SM utilisation:           0.87%
//   - Warp cycles per issue:  775 (ideal: ~4-8)

__global__ void find_aabbs_naive(
    const int* __restrict__ seg,   // [h x w] object IDs, uploaded as int
    int* min_x, int* min_y,        // [MAX_IDS] lower bounds
    int* max_x, int* max_y,        // [MAX_IDS] upper bounds
    int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int id = seg[y * w + x];
    if (id == 0) return;  // background - no AABB to update

    atomicMin(&min_x[id], x);
    atomicMin(&min_y[id], y);
    atomicMax(&max_x[id], x);
    atomicMax(&max_y[id], y);
}

// --- Kernel 2: per-block shared-memory reduction -----------------------------
//
// Three phases:
//
//   Phase 1 -- initialise shared memory.
//     Each thread initialises one entry of each shared array (stride loop so
//     correctness does not depend on block size == MAX_IDS).
//
//   Phase 2 -- reduce into shared memory.
//     Each thread updates s_min/s_max with shared-memory atomics (low latency,
//     on-chip, no LG queue pressure). Simultaneously, the first thread to
//     encounter each ID registers it in s_unique_ids via atomicExch, ensuring
//     exactly one entry per unique ID with no duplicates.
//
//   Phase 3 -- flush to global memory.
//     Only s_unique_count threads execute -- one per unique ID found in this
//     block. Each issues exactly four global atomics for its ID.
//     Global atomic count per block: s_unique_count * 4 (typically 4-16)
//     instead of up to block_size * 4 (up to 1024) in the naive version.
//
// Shared memory per block: 6 * MAX_IDS * 4 bytes = 6 KB.
// Occupancy impact: none -- warp count is the binding constraint, not smem.

__global__ void find_aabbs_shared(
    const int* __restrict__ seg,   // [h x w] object IDs, uploaded as int
    int* min_x, int* min_y,        // [MAX_IDS] lower bounds
    int* max_x, int* max_y,        // [MAX_IDS] upper bounds
    int w, int h)
{
    __shared__ int s_min_x[MAX_IDS];
    __shared__ int s_min_y[MAX_IDS];
    __shared__ int s_max_x[MAX_IDS];
    __shared__ int s_max_y[MAX_IDS];
    __shared__ int s_seen[MAX_IDS];        // 0 = ID not yet seen in this block
    __shared__ int s_unique_ids[MAX_IDS];  // compact list of IDs seen
    __shared__ int s_unique_count;         // length of s_unique_ids

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;

    // -- phase 1: initialise shared memory ------------------------------------
    for (int i = tid; i < MAX_IDS; i += block_size) {
        s_min_x[i] = INT_MAX;
        s_min_y[i] = INT_MAX;
        s_max_x[i] = INT_MIN;
        s_max_y[i] = INT_MIN;
        s_seen[i]  = 0;
    }
    if (tid == 0) s_unique_count = 0;
    __syncthreads();

    // -- phase 2: reduce into shared memory -----------------------------------
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int id = seg[y * w + x];
        if (id != 0) {
            atomicMin(&s_min_x[id], x);
            atomicMin(&s_min_y[id], y);
            atomicMax(&s_max_x[id], x);
            atomicMax(&s_max_y[id], y);

            // Register this ID exactly once per block.
            // atomicExch returns the old value; if it was 0 this thread wins
            // the race and claims the slot -- all later threads see 1 and skip.
            if (atomicExch(&s_seen[id], 1) == 0) {
                int slot = atomicAdd(&s_unique_count, 1);
                s_unique_ids[slot] = id;
            }
        }
    }
    __syncthreads();

    // -- phase 3: flush local results to global memory ------------------------
    // One thread per unique ID -- typically only 1-4 threads do any work here.
    if (tid < s_unique_count) {
        int id = s_unique_ids[tid];
        atomicMin(&min_x[id], s_min_x[id]);
        atomicMin(&min_y[id], s_min_y[id]);
        atomicMax(&max_x[id], s_max_x[id]);
        atomicMax(&max_y[id], s_max_y[id]);
    }
}

// --- Kernel 3: 2x2 tile with register-level run accumulation -----------------
//
// Each thread owns a 2x2 tile of pixels.  It scans the four pixels in row-major
// order and tracks a "current run": consecutive pixels with the same non-zero ID.
// Bounds are accumulated in registers; a commit to shared memory only happens
// when the ID changes (or the tile ends).
//
// Best case (uniform tile, one object): 1 shared-memory commit per thread
//   instead of 4 -- a 4x reduction in MIO queue pressure for phase 2.
// Worst case (all four pixels different IDs): same as find_aabbs_shared.
// Typical case (spatially coherent objects): close to best case.
//
// Grid is ceil(w/32) x ceil(h/32) -- 4x fewer blocks than find_aabbs_shared --
// so phase-1 initialisation overhead (the five shared-memory array writes per
// block) also drops by 4x.
//
// Profiler expectations (vs find_aabbs_shared at 142 us):
//   - MIO Throttle should drop from ~45% toward ~10-15%
//   - Phase-1 init cost drops from 5.24M to ~1.31M shared-memory writes
//   - Kernel duration target: ~30-60 us (2-5x improvement)

__global__ void find_aabbs_tiled(
    const int* __restrict__ seg,   // [h x w] object IDs, uploaded as int
    int* min_x, int* min_y,        // [MAX_IDS] lower bounds
    int* max_x, int* max_y,        // [MAX_IDS] upper bounds
    int w, int h)
{
    __shared__ int s_min_x[MAX_IDS];
    __shared__ int s_min_y[MAX_IDS];
    __shared__ int s_max_x[MAX_IDS];
    __shared__ int s_max_y[MAX_IDS];
    __shared__ int s_seen[MAX_IDS];        // 0 = ID not yet seen in this block
    __shared__ int s_unique_ids[MAX_IDS];  // compact list of IDs seen
    __shared__ int s_unique_count;         // length of s_unique_ids

    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // -- phase 1: initialise shared memory ------------------------------------
    for (int i = tid; i < MAX_IDS; i += block_size) {
        s_min_x[i] = INT_MAX;
        s_min_y[i] = INT_MAX;
        s_max_x[i] = INT_MIN;
        s_max_y[i] = INT_MIN;
        s_seen[i]  = 0;
    }
    if (tid == 0) s_unique_count = 0;
    __syncthreads();

    // -- phase 2: scan 2x2 tile, committing to shared on ID change ------------
    //
    // This thread's tile top-left pixel:
    const int px = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int py = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    // Flush one accumulated run to shared memory.
    // Macro avoids a function call; the compiler will inline it at each site.
#define COMMIT(id, mnx, mny, mxx, mxy)                         \
    do {                                                        \
        atomicMin(&s_min_x[id], mnx);                          \
        atomicMin(&s_min_y[id], mny);                          \
        atomicMax(&s_max_x[id], mxx);                          \
        atomicMax(&s_max_y[id], mxy);                          \
        if (atomicExch(&s_seen[id], 1) == 0) {                 \
            int _slot = atomicAdd(&s_unique_count, 1);         \
            s_unique_ids[_slot] = id;                          \
        }                                                       \
    } while (0)

    // Process one pixel (cx, cy) and update the running accumulation.
    // cur_id == 0 means "no active run".
#define PROCESS_PIXEL(dx, dy)                                          \
    do {                                                               \
        const int cx = px + (dx);                                      \
        const int cy = py + (dy);                                      \
        const int id = (cx < w && cy < h) ? seg[cy * w + cx] : 0;    \
        if (id != 0 && id == cur_id) {                                 \
            /* Extend current run -- no atomic, pure register work */  \
            if (cx < lmnx) lmnx = cx;                                  \
            if (cy < lmny) lmny = cy;                                  \
            if (cx > lmxx) lmxx = cx;                                  \
            if (cy > lmxy) lmxy = cy;                                  \
        } else {                                                       \
            if (cur_id != 0) COMMIT(cur_id, lmnx, lmny, lmxx, lmxy); \
            cur_id = id;                                               \
            lmnx = cx; lmny = cy; lmxx = cx; lmxy = cy;               \
        }                                                              \
    } while (0)

    int cur_id = 0;              // 0 = no active run
    int lmnx, lmny, lmxx, lmxy; // register-level bounds for current run

    PROCESS_PIXEL(0, 0);
    PROCESS_PIXEL(1, 0);
    PROCESS_PIXEL(0, 1);
    PROCESS_PIXEL(1, 1);

    // Flush the final run (if the tile ended mid-run).
    if (cur_id != 0) COMMIT(cur_id, lmnx, lmny, lmxx, lmxy);

#undef PROCESS_PIXEL
#undef COMMIT

    __syncthreads();

    // -- phase 3: flush local results to global memory ------------------------
    if (tid < s_unique_count) {
        const int id = s_unique_ids[tid];
        atomicMin(&min_x[id], s_min_x[id]);
        atomicMin(&min_y[id], s_min_y[id]);
        atomicMax(&max_x[id], s_max_x[id]);
        atomicMax(&max_y[id], s_max_y[id]);
    }
}

// --- Main --------------------------------------------------------------------

int main(int argc, char* argv[])
{
    const char* seg_path = (argc > 1) ? argv[1] : DEFAULT_SEG_PATH;

    // -- 1. Load segmentation PNG ---------------------------------------------
    //
    // stbi_load with req_comp=1 forces a single-channel (grayscale) decode
    // regardless of what channels are stored in the file.
    int w, h, file_channels;
    uint8_t* raw = stbi_load(seg_path, &w, &h, &file_channels, 1);
    if (!raw) {
        fprintf(stderr, "Failed to load '%s': %s\n",
                seg_path, stbi_failure_reason());
        return 1;
    }
    printf("Loaded %s  (%dx%d)\n", seg_path, w, h);

    // Convert uint8_t IDs to int for the kernel.
    // Note: storing IDs as int rather than uint8_t wastes 4x memory bandwidth --
    // a straightforward optimisation to explore later.
    std::vector<int> seg_int(w * h);
    for (int i = 0; i < w * h; ++i) seg_int[i] = raw[i];
    stbi_image_free(raw);

    // -- 2. Upload segmentation to the GPU ------------------------------------
    int* d_seg;
    CUDA_CHECK(cudaMalloc(&d_seg, w * h * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_seg, seg_int.data(),
                          w * h * sizeof(int), cudaMemcpyHostToDevice));

    // -- 3. Allocate per-ID bound arrays on the GPU ---------------------------
    int *d_min_x, *d_min_y, *d_max_x, *d_max_y;
    CUDA_CHECK(cudaMalloc(&d_min_x, MAX_IDS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_y, MAX_IDS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_x, MAX_IDS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_y, MAX_IDS * sizeof(int)));

    // Sentinels: min starts at INT_MAX, max starts at INT_MIN.
    auto reset_bounds = [&]() {
        std::vector<int> v_min(MAX_IDS, INT_MAX);
        std::vector<int> v_max(MAX_IDS, INT_MIN);
        CUDA_CHECK(cudaMemcpy(d_min_x, v_min.data(), MAX_IDS*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_min_y, v_min.data(), MAX_IDS*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_max_x, v_max.data(), MAX_IDS*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_max_y, v_max.data(), MAX_IDS*sizeof(int), cudaMemcpyHostToDevice));
    };

    // -- 4. Configure launch --------------------------------------------------
    //
    // 16x16 = 256 threads/block is a common starting point for 2-D kernels.
    dim3 block(16, 16);

    // find_aabbs_naive / find_aabbs_shared: one thread per pixel.
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    // find_aabbs_tiled: one thread per 2x2 tile.
    // ceil(w/2) tiles along x, ceil(h/2) along y; then ceil those by block dim.
    dim3 grid_tiled(((w + 1) / 2 + block.x - 1) / block.x,
                    ((h + 1) / 2 + block.y - 1) / block.y);

    printf("Launch config (naive/shared): grid(%u, %u)  block(%u, %u)  = %u threads\n",
           grid.x, grid.y, block.x, block.y,
           grid.x * grid.y * block.x * block.y);
    printf("Launch config (tiled):        grid(%u, %u)  block(%u, %u)  = %u threads\n",
           grid_tiled.x, grid_tiled.y, block.x, block.y,
           grid_tiled.x * grid_tiled.y * block.x * block.y);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Helper: warm up then time a kernel launch, leaving results in the bound arrays.
    auto time_kernel = [&](auto kernel_fn, const char* label) -> float {
        // Warm-up -- absorbs driver init and JIT overhead.
        reset_bounds();
        kernel_fn();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed run.
        reset_bounds();
        CUDA_CHECK(cudaEventRecord(ev_start));
        kernel_fn();
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        printf("%-30s  %.4f ms\n", label, ms);
        return ms;
    };

    // -- 5. Run and time all three kernels ------------------------------------
    printf("\n%-30s  %s\n", "Kernel", "Time");
    printf("----------------------------------------------\n");

    time_kernel([&]() {
        find_aabbs_naive<<<grid, block>>>(d_seg, d_min_x, d_min_y, d_max_x, d_max_y, w, h);
    }, "find_aabbs_naive");

    // Snapshot naive results before resetting for the next kernel.
    std::vector<int> ref_min_x(MAX_IDS), ref_min_y(MAX_IDS);
    std::vector<int> ref_max_x(MAX_IDS), ref_max_y(MAX_IDS);
    CUDA_CHECK(cudaMemcpy(ref_min_x.data(), d_min_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_min_y.data(), d_min_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_max_x.data(), d_max_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_max_y.data(), d_max_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));

    time_kernel([&]() {
        find_aabbs_shared<<<grid, block>>>(d_seg, d_min_x, d_min_y, d_max_x, d_max_y, w, h);
    }, "find_aabbs_shared");

    // Correctness check: find_aabbs_shared vs naive reference.
    {
        std::vector<int> t_min_x(MAX_IDS), t_min_y(MAX_IDS);
        std::vector<int> t_max_x(MAX_IDS), t_max_y(MAX_IDS);
        CUDA_CHECK(cudaMemcpy(t_min_x.data(), d_min_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(t_min_y.data(), d_min_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(t_max_x.data(), d_max_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(t_max_y.data(), d_max_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        int mismatches = 0;
        for (int id = 1; id < MAX_IDS; ++id) {
            if (t_min_x[id] != ref_min_x[id] || t_min_y[id] != ref_min_y[id] ||
                t_max_x[id] != ref_max_x[id] || t_max_y[id] != ref_max_y[id]) {
                fprintf(stderr, "MISMATCH id=%d  naive=(%d,%d,%d,%d)  shared=(%d,%d,%d,%d)\n",
                        id,
                        ref_min_x[id], ref_min_y[id], ref_max_x[id], ref_max_y[id],
                        t_min_x[id],   t_min_y[id],   t_max_x[id],   t_max_y[id]);
                ++mismatches;
            }
        }
        if (mismatches == 0)
            printf("Correctness check: PASS (find_aabbs_shared matches naive)\n");
        else
            fprintf(stderr, "Correctness check: FAIL (%d mismatches)\n", mismatches);
    }

    time_kernel([&]() {
        find_aabbs_tiled<<<grid_tiled, block>>>(d_seg, d_min_x, d_min_y, d_max_x, d_max_y, w, h);
    }, "find_aabbs_tiled");

    // Correctness check: find_aabbs_tiled vs naive reference.
    {
        std::vector<int> t_min_x(MAX_IDS), t_min_y(MAX_IDS);
        std::vector<int> t_max_x(MAX_IDS), t_max_y(MAX_IDS);
        CUDA_CHECK(cudaMemcpy(t_min_x.data(), d_min_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(t_min_y.data(), d_min_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(t_max_x.data(), d_max_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(t_max_y.data(), d_max_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
        int mismatches = 0;
        for (int id = 1; id < MAX_IDS; ++id) {
            if (t_min_x[id] != ref_min_x[id] || t_min_y[id] != ref_min_y[id] ||
                t_max_x[id] != ref_max_x[id] || t_max_y[id] != ref_max_y[id]) {
                fprintf(stderr, "MISMATCH id=%d  naive=(%d,%d,%d,%d)  tiled=(%d,%d,%d,%d)\n",
                        id,
                        ref_min_x[id], ref_min_y[id], ref_max_x[id], ref_max_y[id],
                        t_min_x[id],   t_min_y[id],   t_max_x[id],   t_max_y[id]);
                ++mismatches;
            }
        }
        if (mismatches == 0)
            printf("Correctness check: PASS (find_aabbs_tiled matches naive)\n");
        else
            fprintf(stderr, "Correctness check: FAIL (%d mismatches)\n", mismatches);
    }

    // -- 6. Download and print results ----------------------------------------
    // Results in the bound arrays are from the last kernel (find_aabbs_tiled).
    std::vector<int> h_min_x(MAX_IDS), h_min_y(MAX_IDS);
    std::vector<int> h_max_x(MAX_IDS), h_max_y(MAX_IDS);
    CUDA_CHECK(cudaMemcpy(h_min_x.data(), d_min_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_y.data(), d_min_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_x.data(), d_max_x, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_y.data(), d_max_y, MAX_IDS*sizeof(int), cudaMemcpyDeviceToHost));

    printf("\nID    x0    y0    x1    y1   width  height\n");
    printf("---------------------------------------------\n");
    for (int id = 1; id < MAX_IDS; ++id) {
        if (h_min_x[id] == INT_MAX) continue;
        printf("[%3d]  %4d  %4d  %4d  %4d   %4d   %4d\n",
               id,
               h_min_x[id], h_min_y[id], h_max_x[id], h_max_y[id],
               h_max_x[id] - h_min_x[id] + 1,
               h_max_y[id] - h_min_y[id] + 1);
    }

    // -- 7. Save result image with bounding-box overlays ----------------------
    auto rgb_result = colourise(seg_int, w, h);
    for (int id = 1; id < MAX_IDS; ++id) {
        if (h_min_x[id] == INT_MAX) continue;
        draw_rect(rgb_result,
                  h_min_x[id], h_min_y[id], h_max_x[id], h_max_y[id],
                  255, 255, 255,  // white outline
                  w, h);
    }
    if (!stbi_write_png("result.png", w, h, 3, rgb_result.data(), w * 3)) {
        fprintf(stderr, "Failed to write result.png\n");
    } else {
        printf("Saved result.png\n");
    }

    // -- 8. Clean up ----------------------------------------------------------
    cudaFree(d_seg);
    cudaFree(d_min_x); cudaFree(d_min_y);
    cudaFree(d_max_x); cudaFree(d_max_y);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
