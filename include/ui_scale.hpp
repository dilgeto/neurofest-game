#pragma once

// Shared UI scale factor: every executable's main() sets this once, right after computing
// its own SCREEN_WIDTH, to SCREEN_WIDTH / 1680.0f -- 1680 being the original design-reference
// width every hardcoded font size, logo height, and button dimension in this codebase was
// tuned against. Anything that draws text or a fixed-size UI element multiplies by this so
// it stays visually proportionate when the window resolution changes, instead of staying
// pinned at its original pixel size on a much bigger (or smaller) canvas.
//
// A C++17 inline variable: exactly one definition across the whole program regardless of how
// many translation units include this header, so no matching .cpp/CMakeLists.txt change is
// needed to share it (unlike a plain `extern` global).
inline float g_uiScale = 1.0f;
