#pragma once
#include <cstdint>

// One RGB entry per object ID. Index 0 = background.
static constexpr int PALETTE_SIZE = 21;

static constexpr uint8_t PALETTE[PALETTE_SIZE][3] = {
    { 20,  20,  20},   // 0  background
    {230,  25,  75},   // 1  red
    { 60, 180,  75},   // 2  green
    {255, 225,  25},   // 3  yellow
    {  0, 130, 200},   // 4  blue
    {245, 130,  48},   // 5  orange
    {145,  30, 180},   // 6  purple
    { 70, 240, 240},   // 7  cyan
    {240,  50, 230},   // 8  magenta
    {210, 245,  60},   // 9  lime
    {250, 190, 212},   // 10 pink
    {  0, 128, 128},   // 11 teal
    {220, 190, 255},   // 12 lavender
    {170, 110,  40},   // 13 brown
    {255, 250, 200},   // 14 beige
    {128,   0,   0},   // 15 maroon
    {  0,   0, 128},   // 16 navy
    {128, 128,   0},   // 17 olive
    {  0, 128,   0},   // 18 dark green
    {128,   0, 128},   // 19 dark magenta
    { 64, 175, 220},   // 20 sky blue
};
