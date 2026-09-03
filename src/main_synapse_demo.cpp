#include <raylib.h>

#include "../include/branding.hpp"
#include "../include/synapse_module.hpp"
#include "../include/ui_scale.hpp"

// Standalone demo (separate binary from NeuroGame): thin wrapper around SynapseModule
// (src/synapse_module.cpp), which holds the actual logic and rendering so it can also run
// inside MultiDemo's combined view. See synapse_module.hpp for what this demo shows.

namespace {
    // Full 3840x2160 monitor, rotated to portrait -- fills it via the borderless-window
    // setup below.
    constexpr int SCREEN_WIDTH = 2160;
    constexpr int SCREEN_HEIGHT = 3840;
}

int main() {
    g_uiScale = SCREEN_WIDTH / 1680.0f;

    // Borderless window sized to exactly fill a monitor, rather than FLAG_FULLSCREEN_MODE:
    // that flag hands sizing to GLFW's "closest video mode on the primary monitor" logic,
    // which on a multi-monitor/scaled setup can pick the wrong monitor or a mismatched
    // resolution entirely (observed firsthand: it shrank the window instead of filling the
    // screen). Undecorated + positioned at the target monitor's origin has no such guesswork.
    SetConfigFlags(FLAG_WINDOW_UNDECORATED);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Transporte de spikes: sinapsis excitatoria vs inhibitoria");
    int targetMonitor = 0;
    for (int m = 0; m < GetMonitorCount(); ++m) {
        if (GetMonitorWidth(m) == SCREEN_WIDTH && GetMonitorHeight(m) == SCREEN_HEIGHT) { targetMonitor = m; break; }
    }
    Vector2 targetMonitorPos = GetMonitorPosition(targetMonitor);
    SetWindowPosition(static_cast<int>(targetMonitorPos.x), static_cast<int>(targetMonitorPos.y));
    SetTargetFPS(60);

    SynapseModule demo;
    // Reserve room at the bottom for DrawSponsorLogos/DrawFondecytCredit (so the module's own
    // bottom-anchored content doesn't render underneath them) and a little padding at the top
    // so nothing touches that edge either.
    demo.setBounds({
        0, ModuleTopPadding(), static_cast<float>(SCREEN_WIDTH),
        static_cast<float>(SCREEN_HEIGHT) - BrandingFooterHeight() - ModuleTopPadding()
    });

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        float frameMs = static_cast<float>(GetFrameTime() * 1000.0);
        demo.update(mouse, frameMs);

        BeginDrawing();
        ClearBackground(RAYWHITE);
        demo.draw(mouse);
        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
