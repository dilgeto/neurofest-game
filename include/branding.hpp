#pragma once

// Draws the ANID / DIINF / FONDECYT sponsor logos anchored to the bottom-right corner of a
// `screenWidth` x `screenHeight` window, scaled down to a small fixed height. Call once per
// frame, after everything else, right before EndDrawing() -- textures are loaded lazily (on
// first call, once the window/GL context exists) and cached for the process's lifetime.
// Paths are relative to the working directory the executable is run from (img/*, same
// convention main.cpp uses for "models/..."), so run built executables from the repo root.
void DrawSponsorLogos(float screenWidth, float screenHeight);

// Draws the funding-project attribution ("Fondecyt Regular 1251455 NeuroMetaEvo: ...")
// anchored to the bottom-left corner, mirroring DrawSponsorLogos on the opposite side. Call
// once per frame alongside it, right before EndDrawing().
void DrawFondecytCredit(float screenWidth, float screenHeight);

// Vertical space (in the same g_uiScale-scaled pixels DrawSponsorLogos/DrawFondecytCredit
// already draw in) that the footer occupies at the bottom of the window -- the taller of the
// logo row and the Fondecyt text block, plus its margin. Callers laying out their own
// content (panel heights, bottom-anchored buttons) should reserve at least this much so
// nothing renders underneath/collides with the footer. Only meaningful after g_uiScale has
// been set for the current window.
float BrandingFooterHeight();
