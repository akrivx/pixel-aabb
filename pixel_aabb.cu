// pixel_aabb.cu
//
// Naive GPU pixel-AABB finder.
//
// Loads a grayscale segmentation PNG (pixel value = object ID, 0 = background),
// then launches one CUDA thread per pixel. Each thread atomically widens the
// AABB for its object in four global arrays (min_x, min_y, max_x, max_y).
//
// Usage:  pixel_aabb [path/to/segmentation.png]
//         Defaults to assets/segmentation.png relative to the source tree (absolute
//         path baked in at build time via DEFAULT_SEG_PATH compile definition).
//
// Outputs:
//   result.png  -- colourised segmentation with white bounding-box overlays
//
// This deliberately simple baseline is the starting point for a
// profiler-driven optimisation exercise using Nsight Systems / Compute.

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

// --- CUDA kernel -------------------------------------------------------------
//
// Naive baseline:
//   - One thread per pixel.
//   - Each thread atomically reduces into four global arrays.
//
// Known bottlenecks (to be revealed by the profiler):
//   - Heavy atomic contention -- all threads for the same object race on the
//     same four memory locations in global memory.
//   - No spatial locality exploitation -- nearby threads that belong to the
//     same object could instead reduce locally (e.g. per-block) first.
//   - The segmentation is stored as int (4 bytes/pixel) even though IDs fit
//     in a uint8_t (1 byte/pixel), wasting 4x global memory bandwidth on reads.

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
    // The profiler will tell us whether occupancy is the binding constraint.
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    printf("Launch config: grid(%u, %u)  block(%u, %u)  = %u total threads\n",
           grid.x, grid.y, block.x, block.y,
           grid.x * grid.y * block.x * block.y);

    // -- 5. Warm-up run (excluded from timing) --------------------------------
    //
    // Absorbs driver init, JIT compilation, and any first-launch overhead so
    // the timed run below reflects steady-state kernel performance.
    reset_bounds();
    find_aabbs_naive<<<grid, block>>>(d_seg, d_min_x, d_min_y, d_max_x, d_max_y, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    // -- 6. Timed run ---------------------------------------------------------
    reset_bounds();

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    find_aabbs_naive<<<grid, block>>>(d_seg, d_min_x, d_min_y, d_max_x, d_max_y, w, h);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float kernel_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    printf("Kernel time: %.4f ms\n", kernel_ms);

    // -- 7. Download and print results ----------------------------------------
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

    // -- 8. Save result image with bounding-box overlays ----------------------
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

    // -- 9. Clean up ----------------------------------------------------------
    cudaFree(d_seg);
    cudaFree(d_min_x); cudaFree(d_min_y);
    cudaFree(d_max_x); cudaFree(d_max_y);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
