// stb_impl.cpp
//
// Exactly one translation unit must define the STB implementations before
// including the headers. All other files include the headers without the
// define, and the linker resolves the symbols from this object file.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
