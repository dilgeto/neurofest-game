#pragma once

// Draws the ANID / DIINF / FONDECYT sponsor logos anchored to the bottom-right corner of a
// `screenWidth` x `screenHeight` window, scaled down to a small fixed height. Call once per
// frame, after everything else, right before EndDrawing() -- textures are loaded lazily (on
// first call, once the window/GL context exists) and cached for the process's lifetime.
// Paths are relative to the working directory the executable is run from (img/*, same
// convention main.cpp uses for "models/..."), so run built executables from the repo root.
void DrawSponsorLogos(float screenWidth, float screenHeight);
