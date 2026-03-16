# pixel-aabb

A CUDA C++ exercise in profiler-driven GPU optimisation.

The problem: given a segmented image where each pixel stores an integer object
ID, find the axis-aligned bounding box (AABB) of every object.

The repository starts with a deliberately naive implementation — one thread per
pixel, four global atomics per non-background pixel — and is intended to be
progressively optimised using **Nsight Systems** and **Nsight Compute**, with
each iteration guided by profiler data.

## Repository layout

```
pixel-aabb/
  generate_seg.cpp   -- generates the segmentation PNG (run once)
  pixel_aabb.cu      -- CUDA kernel + host driver (the thing to optimise)
  vis.h              -- visualisation helpers (colourise, draw_rect)
  palette.h          -- shared colour palette
  stb_impl.cpp       -- single TU that provides stb_image implementations
  CMakeLists.txt
```

## Prerequisites

| Requirement | Tested version |
|---|---|
| CMake | 3.18 or later |
| CUDA Toolkit | 12.x |
| C++ compiler | MSVC 2022 (Windows) / GCC 11+ (Linux) |
| Git | any recent version (used by FetchContent to download stb) |

`stb_image` and `stb_image_write` are downloaded automatically by CMake on
the first configure — no manual steps required.

## Build

```bash
# From the repo root:
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86   # adjust for your GPU
cmake --build build --config Release
```

GPU architecture values:

| Value | GPU family |
|---|---|
| 75 | Turing (RTX 20xx, GTX 16xx) |
| 86 | Ampere (RTX 30xx) |
| 89 | Ada (RTX 40xx) |
| 90 | Hopper (H100) |

Binaries are written to `build/Release/` on Windows or `build/` on Linux.

## Running the AABB finder

A pre-generated segmentation image is included at `assets/segmentation.png`.
The absolute path to that file is baked into the binary at build time, so the
executable works from any working directory:

```bash
build/Release/pixel_aabb.exe              # Windows
build/pixel_aabb                          # Linux
```

To use a different image, pass the path explicitly:

```bash
./pixel_aabb path/to/other.png
```

`pixel_aabb` accepts any 8-bit grayscale PNG where pixel values are object IDs
(0 = background). It prints a table of AABBs and writes `result.png` —
the colourised segmentation with white bounding-box overlays — to the current
working directory.

## Regenerating or customising the input image

`generate_seg` is only needed if you want to modify the scene (edit `SCENE[]`
in `generate_seg.cpp`) or produce a fresh `segmentation.png`. Run it from the
repo root so the output lands in `assets/`:

```bash
cd assets
../build/Release/generate_seg             # Windows
../build/generate_seg                     # Linux
```

It writes two files:

| File | Description |
|---|---|
| `segmentation.png` | Grayscale PNG — pixel value = object ID. |
| `segmentation_preview.png` | Colourised RGB PNG — open this to inspect the scene. |

Both open natively in Windows Photos (double-click).

Example output:

```
Loaded segmentation.png  (1024x1024)
Launch config: grid(64, 64)  block(16, 16)  = 1048576 total threads
Kernel time: 0.0521 ms

ID    x0    y0    x1    y1   width  height
---------------------------------------------
[  1]    60   90   239   209    180    120
[  2]   270   45   489   194    220    150
...
Saved result.png
```

## Profiling

The kernel is compiled with `-lineinfo` so profiler output maps back to source
lines.

**Nsight Systems** (timeline, CPU/GPU overlap, memory transfers):
```bash
nsys profile --trace=cuda ./pixel_aabb
nsys-ui report1.nsys-rep
```

**Nsight Compute** (per-kernel hardware metrics):
```bash
ncu --set full -o profile ./pixel_aabb
ncu-ui profile.ncu-rep
```

Key metrics to examine on the naive baseline:

- `l2_global_atomic_store_transactions` — reveals the contention cost of the
  global atomics.
- `sm_active_cycles / sm_elapsed_cycles` — low ratio indicates the SMs are
  stalling, likely on atomic serialisation.
- `dram_read_transactions` — baseline reads IDs as `int` (4 bytes/pixel);
  switching to `uint8_t` should reduce this by 4x.

## Planned optimisations

Each step should be validated against the profiler before moving to the next.

1. **Per-block shared-memory reduction** — threads within a block reduce into
   shared memory first, then one thread performs the global atomic. Eliminates
   most contention.
2. **uint8_t input** — store object IDs as one byte per pixel instead of four,
   reducing global memory read traffic by 4x.
3. **Warp-level reduction** — use `__reduce_min_sync` / `__reduce_max_sync`
   (sm_80+) to reduce within a warp before touching shared memory.
4. **Coalescing and layout experiments** — investigate whether a
   structure-of-arrays layout for the bounds or a different thread-block shape
   improves L2 hit rate.
