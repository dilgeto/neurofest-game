#include <raylib.h>

#include "../include/branding.hpp"
#include "../include/ui_scale.hpp"
#include "../include/wann_evolution_module.hpp"

// Standalone demo (separate binary from NeuroGame): thin wrapper around WannEvolutionModule
// (src/wann_evolution_module.cpp), which holds the actual logic and rendering so it can also
// run inside MultiDemo's combined view. See wann_evolution_module.hpp for what this demo
// shows.
//
// Unlike the other demos here, this one intentionally runs at 1920x1080 -- a normal single
// landscape screen, not the festival's dual-4K portrait kiosk -- so it skips the
// borderless/monitor-matching window setup the others use.

namespace {
    constexpr int SCREEN_WIDTH  = 1920;
    constexpr int SCREEN_HEIGHT = 1080;
}

int main() {
    g_uiScale = SCREEN_WIDTH / 1680.0f;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Neuroevolución WANN");
    SetTargetFPS(60);

    WannEvolutionModule demo;
    // Reserve room at the bottom for DrawSponsorLogos/DrawFondecytCredit (so the module's own
    // bottom-anchored controls don't render underneath them) and a little padding at the top
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
