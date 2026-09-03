#pragma once
#include <raylib.h>
#include "ui_scale.hpp"

// Padding reserved at the very top of the window before any module's content starts, so
// nothing touches the top edge either -- the mirror image of BrandingFooterHeight()
// (branding.hpp) reserving room at the bottom for the sponsor logos/Fondecyt credit. Unlike
// that one, there's no footer content to fit up here, so this is just a fixed breathing-room
// constant rather than something computed from what's drawn.
inline float ModuleTopPadding() { return 30.0f * g_uiScale; }

// Common interface every extracted demo (Izhikevich, Synapse, Encoder, EncoderWANN, Decoder)
// implements, so both its standalone main_*.cpp wrapper (full-screen bounds) and
// main_multi_demo.cpp (a fraction of the window's height, stacked with others) can drive it
// the same way:
//   setBounds() assigns the Rectangle this module renders into -- always the window's full
//   width, but only a share of its height when shown alongside other modules. Call it once
//   up front and again any time the assignment changes (e.g. the user picks a different
//   combination in the selector).
//   update() advances one frame's worth of input handling + simulation.
//   draw() renders the module's current state within its last-assigned bounds.
// A module still reads GetMousePosition() itself (via the `mouse` passed to update()) rather
// than being told "you have focus" -- since panels are stacked non-overlapping rectangles,
// the mouse naturally only lands on the widgets of whichever panel it's actually over.
class IDemoModule {
public:
    virtual ~IDemoModule() = default;
    virtual const char* name() const = 0;
    virtual void setBounds(Rectangle bounds) = 0;
    virtual void update(Vector2 mouse, float frameMs) = 0;
    virtual void draw(Vector2 mouse) const = 0;
};
