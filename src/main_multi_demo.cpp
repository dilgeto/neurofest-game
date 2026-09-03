#include <raylib.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "../include/branding.hpp"
#include "../include/button.hpp"
#include "../include/decoder_module.hpp"
#include "../include/demo_module.hpp"
#include "../include/encoder_module.hpp"
#include "../include/encoder_wann_module.hpp"
#include "../include/gamepad_setup.hpp"
#include "../include/izhikevich_module.hpp"
#include "../include/network_view_module.hpp"
#include "../include/synapse_module.hpp"
#include "../include/ui_scale.hpp"
#include "../include/vs_ai_discrete_module.hpp"
#include "../include/vs_ai_iris_module.hpp"
#include "../include/vs_ai_racing_car_module.hpp"

// Combines 2 or 3 modules into one window at once, stacked vertically -- each module keeps
// the window's full width, only its share of the height changes (see include/demo_module.hpp
// for why that's the layout that lets every module's existing text/panel sizing "just work"
// without per-panel rescaling). A small selector screen picks which ones.
//
// Two selection groups, with different rules:
//   Regular (indices 0-7): the 5 standalone demos plus "Evaluar red" for each of the 3
//     tasks. Freely multi-selectable, 2-3 at a time, like before.
//   VS IA (indices 8-11): at most ONE of these four. Choosing one caps the total selection
//     at exactly 2 (itself + exactly one regular demo) and always renders on top -- VS IA's
//     human-vs-AI panels are already denser (two sub-panels + controls) than the other demos,
//     so pairing it with more than one neighbor would cramp everything.

namespace {
    // Full 3840x2160 monitor, rotated to portrait -- fills it via the borderless-window
    // setup below.
    constexpr int SCREEN_WIDTH = 2160;
    constexpr int SCREEN_HEIGHT = 3840;

    constexpr int REGULAR_COUNT = 8;
    constexpr int VS_AI_START = 8;
    constexpr int VS_AI_COUNT = 4;
    constexpr int DEMO_COUNT = REGULAR_COUNT + VS_AI_COUNT;

    const char* const DEMO_LABELS[DEMO_COUNT] = {
        "Patrones de Izhikevich",
        "Transporte de spikes",
        "Codificadores (genérico)",
        "Codificadores WANN",
        "Decodificadores",
        "Evaluar red: Acrobot",
        "Evaluar red: Mountain Car",
        "Evaluar red: Racing Car",
        "VS IA: Acrobot",
        "VS IA: Mountain Car",
        "VS IA: Racing Car",
        "VS IA: Iris",
    };

    std::unique_ptr<IDemoModule> createModule(int index) {
        switch (index) {
            case 0: return std::make_unique<IzhikevichModule>();
            case 1: return std::make_unique<SynapseModule>();
            case 2: return std::make_unique<EncoderModule>();
            case 3: return std::make_unique<EncoderWannModule>();
            case 4: return std::make_unique<DecoderModule>();
            case 5: return std::make_unique<NetworkViewModule>(0);
            case 6: return std::make_unique<NetworkViewModule>(1);
            case 7: return std::make_unique<NetworkViewModule>(2);
            case 8: return std::make_unique<VsAiDiscreteModule>(0);
            case 9: return std::make_unique<VsAiDiscreteModule>(1);
            case 10: return std::make_unique<VsAiRacingCarModule>();
            case 11: return std::make_unique<VsAiIrisModule>();
            default: return nullptr;
        }
    }

    enum class Screen { SELECTOR, COMBINED };
}

int main() {
    g_uiScale = SCREEN_WIDTH / 1680.0f;

    // Borderless window sized to exactly fill a monitor, rather than FLAG_FULLSCREEN_MODE:
    // that flag hands sizing to GLFW's "closest video mode on the primary monitor" logic,
    // which on a multi-monitor/scaled setup can pick the wrong monitor or a mismatched
    // resolution entirely (observed firsthand: it shrank the window instead of filling the
    // screen). Undecorated + positioned at the target monitor's origin has no such guesswork.
    SetConfigFlags(FLAG_WINDOW_UNDECORATED);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Varias demos a la vez");
    int targetMonitor = 0;
    for (int m = 0; m < GetMonitorCount(); ++m) {
        if (GetMonitorWidth(m) == SCREEN_WIDTH && GetMonitorHeight(m) == SCREEN_HEIGHT) { targetMonitor = m; break; }
    }
    Vector2 targetMonitorPos = GetMonitorPosition(targetMonitor);
    SetWindowPosition(static_cast<int>(targetMonitorPos.x), static_cast<int>(targetMonitorPos.y));
    SetTargetFPS(60);

    // See gamepad_setup.cpp for why this specific controller needs a hand-derived mapping,
    // and why it must be called (once, right here) in every binary that reads gamepad input
    // -- VsAiDiscreteModule and VsAiRacingCarModule both do.
    SetupGameSirGamepadMapping();

    Screen screen = Screen::SELECTOR;

    // Selector: one toggle button per demo, in two stacked groups (regular, then VS IA),
    // plus "Comenzar" and a status line explaining why it's disabled when it is.
    constexpr float TOGGLE_WIDTH = 700.0f;
    constexpr float TOGGLE_HEIGHT = 56.0f;
    constexpr float TOGGLE_GAP = 10.0f;
    constexpr float GROUP_GAP = 30.0f; // extra space between the regular and VS IA groups
    float allTogglesHeight = DEMO_COUNT * (TOGGLE_HEIGHT + TOGGLE_GAP) + GROUP_GAP;
    float togglesTop = SCREEN_HEIGHT / 2.0f - allTogglesHeight / 2.0f;

    std::array<Button, DEMO_COUNT> toggleButtons = {
        Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""},
        Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""},
        Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""}, Button{{0,0,1,1}, ""},
    };
    std::array<bool, DEMO_COUNT> selected{};
    for (int i = 0; i < DEMO_COUNT; ++i) {
        float y = togglesTop + i * (TOGGLE_HEIGHT + TOGGLE_GAP) + (i >= VS_AI_START ? GROUP_GAP : 0.0f);
        toggleButtons[static_cast<size_t>(i)].setBounds({ SCREEN_WIDTH / 2.0f - TOGGLE_WIDTH / 2.0f, y, TOGGLE_WIDTH, TOGGLE_HEIGHT });
        toggleButtons[static_cast<size_t>(i)].setText(std::string("[ ] ") + DEMO_LABELS[i]);
        toggleButtons[static_cast<size_t>(i)].setFontSize(static_cast<int>(24 * g_uiScale));
    }
    float vsAiHeaderY = togglesTop + VS_AI_START * (TOGGLE_HEIGHT + TOGGLE_GAP) + GROUP_GAP / 2.0f - 12.0f;

    float startButtonTop = togglesTop + allTogglesHeight + 30.0f;
    Button startButton({ SCREEN_WIDTH / 2.0f - 200.0f, startButtonTop, 400.0f, 70.0f }, "Comenzar");

    // Combined view: whichever modules are currently active, stacked full-width, with a VS
    // IA (if any) always first (top).
    std::vector<std::unique_ptr<IDemoModule>> activeModules;
    Button backButton({ 20, 20, 140, 40 }, "Volver");

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        int regularCount = 0;
        for (int i = 0; i < REGULAR_COUNT; ++i) regularCount += selected[static_cast<size_t>(i)] ? 1 : 0;
        int vsAiSelected = -1;
        for (int i = VS_AI_START; i < DEMO_COUNT; ++i) if (selected[static_cast<size_t>(i)]) vsAiSelected = i;

        bool canStart = (vsAiSelected >= 0) ? (regularCount == 1) : (regularCount >= 2 && regularCount <= 3);

        if (screen == Screen::SELECTOR) {
            for (int i = 0; i < DEMO_COUNT; ++i) {
                if (!toggleButtons[static_cast<size_t>(i)].isClicked(mouse)) continue;
                bool nowSelected = !selected[static_cast<size_t>(i)];
                if (i >= VS_AI_START && nowSelected) {
                    // Radio behavior within the VS IA group: selecting one deselects any
                    // other VS IA that was checked.
                    for (int j = VS_AI_START; j < DEMO_COUNT; ++j) {
                        selected[static_cast<size_t>(j)] = false;
                        toggleButtons[static_cast<size_t>(j)].setText(std::string("[ ] ") + DEMO_LABELS[j]);
                    }
                }
                selected[static_cast<size_t>(i)] = nowSelected;
                toggleButtons[static_cast<size_t>(i)].setText(std::string(nowSelected ? "[x] " : "[ ] ") + DEMO_LABELS[i]);
            }
            if (canStart && startButton.isClicked(mouse)) {
                activeModules.clear();
                if (vsAiSelected >= 0) activeModules.push_back(createModule(vsAiSelected)); // always on top
                for (int i = 0; i < REGULAR_COUNT; ++i) {
                    if (selected[static_cast<size_t>(i)]) activeModules.push_back(createModule(i));
                }
                // Reserve room at the bottom for DrawSponsorLogos/DrawFondecytCredit (so the
                // bottom-most panel's own content doesn't render underneath them) and a
                // little padding at the top so the topmost panel doesn't touch that edge.
                float contentTop = ModuleTopPadding();
                float contentHeight = static_cast<float>(SCREEN_HEIGHT) - BrandingFooterHeight() - contentTop;
                float panelHeight = contentHeight / static_cast<float>(activeModules.size());
                for (size_t i = 0; i < activeModules.size(); ++i) {
                    activeModules[i]->setBounds({ 0, contentTop + i * panelHeight, static_cast<float>(SCREEN_WIDTH), panelHeight });
                }
                screen = Screen::COMBINED;
            }
        } else { // COMBINED
            if (backButton.isClicked(mouse)) {
                activeModules.clear(); // drop the running instances; re-picking re-creates them fresh
                screen = Screen::SELECTOR;
            } else {
                float frameMs = static_cast<float>(GetFrameTime() * 1000.0);
                for (auto& module : activeModules) module->update(mouse, frameMs);
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        if (screen == Screen::SELECTOR) {
            const char* title = "Elegí 2 o 3 demos para mostrar juntas";
            int titleWidth = MeasureText(title, static_cast<int>(30 * g_uiScale));
            DrawText(title, SCREEN_WIDTH / 2 - titleWidth / 2, static_cast<int>(togglesTop - 80.0f), static_cast<int>(30 * g_uiScale), DARKGRAY);

            const char* vsAiHeader = "VS IA (a lo sumo 1 -- siempre queda arriba, y limita el total a 2)";
            int vsAiHeaderWidth = MeasureText(vsAiHeader, static_cast<int>(15 * g_uiScale));
            DrawText(vsAiHeader, SCREEN_WIDTH / 2 - vsAiHeaderWidth / 2, static_cast<int>(vsAiHeaderY), static_cast<int>(15 * g_uiScale), GRAY);

            for (Button& b : toggleButtons) b.draw(mouse);

            startButton.draw(mouse);
            if (!canStart) {
                // Dim the button so it visibly reads as disabled even though Button has no
                // built-in disabled state -- a translucent overlay is simpler than adding
                // one to the shared Button class for a single caller.
                DrawRectangleRec(startButton.getButton(), Fade(RAYWHITE, 0.55f));
            }
            std::string status = (vsAiSelected >= 0)
                ? (regularCount == 1 ? "Listo para comenzar" : "Con un VS IA activo, selecciona exactamente 1 demo más")
                : (regularCount < 2 ? "Selecciona al menos 2" : regularCount > 3 ? "Selecciona como máximo 3" : "Listo para comenzar");
            int statusWidth = MeasureText(status.c_str(), static_cast<int>(16 * g_uiScale));
            DrawText(status.c_str(), SCREEN_WIDTH / 2 - statusWidth / 2, static_cast<int>(startButtonTop + 90.0f), static_cast<int>(16 * g_uiScale), GRAY);
        } else {
            for (auto& module : activeModules) module->draw(mouse);
            backButton.draw(mouse);
        }

        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
