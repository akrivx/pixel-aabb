cmake_minimum_required(VERSION 3.18)
project(pixel_aabb LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# --- GPU architecture ---------------------------------------------------------
# Adjust to match your GPU:
#   75 -> Turing  (RTX 20xx, GTX 16xx)
#   86 -> Ampere  (RTX 30xx)
#   89 -> Ada     (RTX 40xx)
#   90 -> Hopper  (H100)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA GPU architecture(s)")

# --- stb (header-only) -------------------------------------------------------
# FetchContent downloads stb on the first configure. No manual steps needed.
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master)

FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

# stb requires exactly one translation unit to define the implementations.
# All other files include the headers without any defines.
add_library(stb_impl STATIC stb_impl.cpp)
target_include_directories(stb_impl PUBLIC ${stb_SOURCE_DIR})

# --- generate_seg: writes segmentation.png + segmentation_preview.png --------
add_executable(generate_seg generate_seg.cpp)
target_link_libraries(generate_seg PRIVATE stb_impl)

# --- pixel_aabb: loads segmentation.png, finds AABBs on GPU, saves result.png -
add_executable(pixel_aabb pixel_aabb.cu)
target_link_libraries(pixel_aabb PRIVATE stb_impl)
# Bake the absolute path to assets/ into the binary so the default input works
# regardless of the working directory the executable is invoked from.
target_compile_definitions(pixel_aabb PRIVATE
    DEFAULT_SEG_PATH="${CMAKE_SOURCE_DIR}/assets/segmentation.png")
target_compile_options(pixel_aabb PRIVATE
    # -lineinfo embeds source-line info without full debug symbols --
    # essential for meaningful Nsight profiler source attribution.
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)
