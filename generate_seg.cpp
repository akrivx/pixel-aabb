// generate_seg.cpp
//
// Generates a synthetic segmented image and writes two files:
//   segmentation.png         -- grayscale, pixel value = object ID (machine-readable)
//   segmentation_preview.png -- colourised RGB, for visual inspection
//
// Run this once. pixel_aabb then loads segmentation.png.

#include "stb_image_write.h"
#include "vis.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static constexpr int IMG_W       = 1024;
static constexpr int IMG_H       = 1024;
static constexpr int NUM_OBJECTS = 20;   // IDs 1..NUM_OBJECTS, 0 = background

// --- Scene definition --------------------------------------------------------
//
// Each object is an axis-aligned ellipse. Where ellipses overlap, the higher
// ID wins (later writes overwrite earlier ones), which is fine - it just
// means some objects will be partially occluded.

struct Ellipse { float cx, cy, rx, ry; int id; };

static constexpr Ellipse SCENE[NUM_OBJECTS] = {
    // cx    cy    rx    ry  id
    { 150,  150,   90,   60,  1},
    { 380,  120,  110,   75,  2},
    { 620,  180,   80,  120,  3},
    { 860,  155,  100,   80,  4},
    { 100,  420,   70,   90,  5},
    { 330,  390,  130,   60,  6},
    { 570,  410,   90,   90,  7},
    { 800,  430,  110,   70,  8},
    { 190,  660,   80,  100,  9},
    { 450,  690,  120,   80, 10},
    { 710,  660,   70,  110, 11},
    { 910,  710,   95,   60, 12},
    { 140,  880,  100,   90, 13},
    { 410,  900,   90,   70, 14},
    { 660,  870,  110,   80, 15},
    { 880,  870,   80,  100, 16},
    { 512,  512,  150,  150, 17},  // large central blob
    { 255,  255,   55,   55, 18},  // small top-left
    { 755,  755,   55,   55, 19},  // small bottom-right
    { 512,  200,   40,   40, 20},  // tiny top-centre
};

// --- Image generation --------------------------------------------------------

// Returns a flat [IMG_H x IMG_W] array where each element is an object ID.
static std::vector<uint8_t> generate_segmentation()
{
    std::vector<uint8_t> seg(IMG_W * IMG_H, 0);

    for (const auto& e : SCENE) {
        int x0 = (int)std::max(0.f,          e.cx - e.rx);
        int x1 = (int)std::min((float)IMG_W, e.cx + e.rx + 1.f);
        int y0 = (int)std::max(0.f,          e.cy - e.ry);
        int y1 = (int)std::min((float)IMG_H, e.cy + e.ry + 1.f);

        for (int y = y0; y < y1; ++y) {
            float dy = (y - e.cy) / e.ry;
            for (int x = x0; x < x1; ++x) {
                float dx = (x - e.cx) / e.rx;
                if (dx*dx + dy*dy <= 1.f)
                    seg[y * IMG_W + x] = (uint8_t)e.id;
            }
        }
    }
    return seg;
}

// --- Main --------------------------------------------------------------------

int main()
{
    printf("Generating %dx%d segmentation with %d objects...\n",
           IMG_W, IMG_H, NUM_OBJECTS);

    auto seg = generate_segmentation();

    // Grayscale PNG: pixel value = object ID. This is the file pixel_aabb reads.
    if (!stbi_write_png("segmentation.png", IMG_W, IMG_H, 1,
                        seg.data(), IMG_W)) {
        fprintf(stderr, "Failed to write segmentation.png\n");
        return 1;
    }
    printf("Saved segmentation.png  (grayscale: pixel value = object ID)\n");

    // Colourised PNG: for visual inspection only.
    auto preview = colourise(seg, IMG_W, IMG_H);
    if (!stbi_write_png("segmentation_preview.png", IMG_W, IMG_H, 3,
                        preview.data(), IMG_W * 3)) {
        fprintf(stderr, "Failed to write segmentation_preview.png\n");
        return 1;
    }
    printf("Saved segmentation_preview.png  (colourised, open this to inspect the scene)\n");

    return 0;
}
