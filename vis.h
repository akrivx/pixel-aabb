#pragma once
// Visualisation helpers shared between generate_seg and pixel_aabb.

#include "palette.h"
#include <cstdint>
#include <vector>

// Map object IDs to palette colours.
// The template accepts both uint8_t (from generate_seg) and int (from pixel_aabb).
template<typename T>
inline std::vector<uint8_t> colourise(const std::vector<T>& seg, int w, int h)
{
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        int id = (int)seg[i];
        if (id < 0 || id >= PALETTE_SIZE) id = 0;
        rgb[3*i+0] = PALETTE[id][0];
        rgb[3*i+1] = PALETTE[id][1];
        rgb[3*i+2] = PALETTE[id][2];
    }
    return rgb;
}

// Draws a 3-pixel-thick rectangle outline for visibility.
inline void draw_rect(std::vector<uint8_t>& rgb,
                      int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b,
                      int w, int h)
{
    auto put = [&](int x, int y) {
        if ((unsigned)x >= (unsigned)w || (unsigned)y >= (unsigned)h) return;
        int i = (y * w + x) * 3;
        rgb[i] = r; rgb[i+1] = g; rgb[i+2] = b;
    };
    for (int t = -1; t <= 1; ++t) {
        for (int x = x0; x <= x1; ++x) { put(x, y0+t); put(x, y1+t); }
        for (int y = y0; y <= y1; ++y) { put(x0+t, y); put(x1+t, y); }
    }
}
