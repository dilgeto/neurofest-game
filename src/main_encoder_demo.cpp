#include <raylib.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "../include/branding.hpp"
#include "../include/ui_scale.hpp"

// Standalone demo (separate binary from NeuroGame): shows how 2 of snn-simulator's spike
// encoders (include/encoding/{ttfs,rate}Encoder.*) turn a single number in [0,1] into a
// spike train -- one temporal encoder, one rate encoder, per the course's request.
// Formulas reimplemented locally (verified against those exact sources) rather than
// linking the external snn-simulator repo:
//   TTFS (linear): one spike, timed so a higher value fires earlier.
//   Rate: a Bernoulli trial each dt with p = value*maxRate/1000*dt.

namespace {
    // Right half of a 3840x2160 monitor.
    constexpr int SCREEN_WIDTH = 2160;
    constexpr int SCREEN_HEIGHT = 3840;

    constexpr double ENCODE_DURATION_MS = 100.0; // window each encoder encodes into
    constexpr double ENCODE_DT_MS = 1.0;
    constexpr double MAX_RATE = 100.0;           // Hz, matches wann-cpp's encoder setup
    constexpr double TTFS_THRESHOLD = 1e-9;

    constexpr double SWEEP_DURATION_MS = 1500.0; // wall-clock time for one cursor pass;
                                                  // stochastic encoders re-roll every pass
    constexpr float FLASH_MAX_MS = 150.0f;

    // --- Encoders (see ttfsEncoder.cpp / rateEncoder.cpp / poissonEncoder.cpp) ---

    std::vector<double> encodeTtfsLinear(double value) {
        if (value < 0.0 || value > 1.0 || value < TTFS_THRESHOLD) return {};
        double tMax = ENCODE_DURATION_MS - ENCODE_DT_MS;
        double t = (1.0 - value) * tMax;
        t = std::round(t / ENCODE_DT_MS) * ENCODE_DT_MS;
        return { std::clamp(t, 0.0, ENCODE_DURATION_MS - ENCODE_DT_MS) };
    }

    std::vector<double> encodeRate(double value, std::mt19937& rng) {
        if (value < 0.0 || value > 1.0) return {};
        std::vector<double> spikes;
        double spikeProb = (value * MAX_RATE / 1000.0) * ENCODE_DT_MS;
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (double t = 0; t < ENCODE_DURATION_MS; t += ENCODE_DT_MS)
            if (dis(rng) < spikeProb) spikes.push_back(t);
        return spikes;
    }

    struct EncoderPanel {
        std::string title;
        std::string formula;
        Color color;
        std::vector<double> spikes;
        std::vector<bool> fired;     // has the sweep cursor already crossed this spike?
        std::vector<float> flashMs;  // brief highlight pulse when the cursor crosses it
    };

    struct Slider {
        Rectangle track;
        float minValue, maxValue, value;
        bool dragging = false;

        void update(Vector2 mouse) {
            Rectangle hitArea = { track.x - 12, track.y - 12, track.width + 24, track.height + 24 };
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, hitArea)) dragging = true;
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragging = false;
            if (dragging) {
                float t = std::clamp((mouse.x - track.x) / track.width, 0.0f, 1.0f);
                value = minValue + t * (maxValue - minValue);
            }
        }

        void draw() const {
            float trackY = track.y + track.height / 2.0f;
            DrawLineEx({ track.x, trackY }, { track.x + track.width, trackY }, 4.0f, LIGHTGRAY);
            float t = (value - minValue) / (maxValue - minValue);
            float handleX = track.x + t * track.width;
            DrawLineEx({ track.x, trackY }, { handleX, trackY }, 4.0f, Color{60, 110, 220, 255});
            DrawCircleV({ handleX, trackY }, 11.0f, Color{60, 110, 220, 255});
            DrawCircleLines(static_cast<int>(handleX), static_cast<int>(trackY), 11.0f, DARKGRAY);
        }
    };

    // Scales a base (1680-wide-reference) pixel size by g_uiScale, rounding to the nearest
    // integer -- use for every DrawText/MeasureText font-size argument.
    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }

    void regenerate(EncoderPanel& panel, std::vector<double> newSpikes) {
        panel.spikes = std::move(newSpikes);
        panel.fired.assign(panel.spikes.size(), false);
        panel.flashMs.assign(panel.spikes.size(), 0.0f);
    }

    void drawEncoderPanel(Rectangle bounds, const EncoderPanel& panel, double cursorTimeMs) {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(panel.title.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), FS(20), DARKGRAY);
        DrawText(panel.formula.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 32), FS(16), GRAY);

        Rectangle plot = { bounds.x + 20, bounds.y + 70, bounds.width - 40, bounds.height - 110 };
        float axisY = plot.y + plot.height * 0.5f;

        DrawLineEx({ plot.x, axisY }, { plot.x + plot.width, axisY }, 1.5f, LIGHTGRAY);
        DrawText("0 ms", static_cast<int>(plot.x), static_cast<int>(plot.y + plot.height + 6), FS(14), GRAY);
        std::string endLabel = TextFormat("%.0f ms", ENCODE_DURATION_MS);
        int endLabelWidth = MeasureText(endLabel.c_str(), FS(14));
        DrawText(endLabel.c_str(), static_cast<int>(plot.x + plot.width) - endLabelWidth, static_cast<int>(plot.y + plot.height + 6), FS(14), GRAY);

        for (size_t i = 0; i < panel.spikes.size(); ++i) {
            float x = plot.x + static_cast<float>(panel.spikes[i] / ENCODE_DURATION_MS) * plot.width;
            float flash = std::clamp(panel.flashMs[i] / FLASH_MAX_MS, 0.0f, 1.0f);
            DrawLineEx({ x, axisY - plot.height * 0.35f }, { x, axisY + plot.height * 0.35f }, 3.0f, panel.color);
            if (flash > 0.0f) DrawCircleV({ x, axisY }, 6.0f + 6.0f * flash, Fade(panel.color, flash * 0.6f));
        }

        // Sweeping cursor: shows the encoding window playing out over (wall-clock) time.
        float cursorX = plot.x + static_cast<float>(cursorTimeMs / ENCODE_DURATION_MS) * plot.width;
        DrawLineEx({ cursorX, plot.y - 6.0f }, { cursorX, plot.y + plot.height + 6.0f }, 2.0f, Fade(DARKGRAY, 0.6f));

        DrawText(TextFormat("%d spikes", static_cast<int>(panel.spikes.size())),
            static_cast<int>(bounds.x + bounds.width - 90), static_cast<int>(bounds.y + 8), FS(16), DARKGRAY);
    }
}

int main() {
    g_uiScale = SCREEN_WIDTH / 1680.0f;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Codificadores: de numero a espigas");
    SetWindowPosition(SCREEN_WIDTH, 0);
    SetTargetFPS(60);
    std::mt19937 rng(std::random_device{}());

    std::vector<EncoderPanel> panels = {
        { "Temporal (TTFS lineal)", "t = (1 - v) x t_max", Color{60, 110, 220, 255}, {}, {}, {} },
        { "Rate", "p(spike) = v x maxRate/1000 x dt, cada dt", Color{60, 170, 90, 255}, {}, {}, {} },
    };

    constexpr float MARGIN_X = 40.0f;
    constexpr float GRID_TOP = 30.0f;
    constexpr float GAP = 20.0f;
    constexpr int COLS = 2;
    constexpr int ROWS = 1;

    Slider valueSlider{ { (SCREEN_WIDTH - 500.0f) / 2.0f, SCREEN_HEIGHT - 90.0f, 500.0f, 10.0f }, 0.0f, 1.0f, 0.5f };
    constexpr float GRID_BOTTOM = SCREEN_HEIGHT - 140.0f;

    float panelWidth = (SCREEN_WIDTH - 2.0f * MARGIN_X - (COLS - 1) * GAP) / COLS;
    float panelHeight = (GRID_BOTTOM - GRID_TOP - (ROWS - 1) * GAP) / ROWS;

    double sweepProgress = 0.0; // 0..1

    // Prime the first draw with a real encoding instead of empty panels.
    regenerate(panels[0], encodeTtfsLinear(valueSlider.value));
    regenerate(panels[1], encodeRate(valueSlider.value, rng));

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        valueSlider.update(mouse);

        float frameMs = static_cast<float>(GetFrameTime() * 1000.0);
        sweepProgress += frameMs / SWEEP_DURATION_MS;
        if (sweepProgress >= 1.0) {
            sweepProgress -= 1.0;
            regenerate(panels[0], encodeTtfsLinear(valueSlider.value));
            regenerate(panels[1], encodeRate(valueSlider.value, rng));
        }
        double cursorTimeMs = sweepProgress * ENCODE_DURATION_MS;

        for (EncoderPanel& panel : panels) {
            for (size_t i = 0; i < panel.spikes.size(); ++i) {
                if (!panel.fired[i] && cursorTimeMs >= panel.spikes[i]) {
                    panel.fired[static_cast<size_t>(i)] = true;
                    panel.flashMs[i] = FLASH_MAX_MS;
                }
                panel.flashMs[i] = std::max(0.0f, panel.flashMs[i] - frameMs);
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        for (int i = 0; i < static_cast<int>(panels.size()); ++i) {
            int row = i / COLS;
            int col = i % COLS;
            Rectangle bounds = {
                MARGIN_X + col * (panelWidth + GAP),
                GRID_TOP + row * (panelHeight + GAP),
                panelWidth, panelHeight
            };
            drawEncoderPanel(bounds, panels[static_cast<size_t>(i)], cursorTimeMs);
        }

        valueSlider.draw();
        DrawText(TextFormat("Valor a codificar: %.2f", valueSlider.value),
            static_cast<int>(valueSlider.track.x), static_cast<int>(valueSlider.track.y - 26), FS(20), DARKGRAY);

        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}