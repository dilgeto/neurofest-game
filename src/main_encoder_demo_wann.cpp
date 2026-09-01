#include <raylib.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "../include/branding.hpp"
#include "../include/snn_network.hpp"

// Standalone demo (separate binary from NeuroGame): shows the two encoders actually used
// by the trained WANN models this project loads (see model_browser.cpp's presets) --
// TTFS (SnnEncoderKind::Ttfs) and SMALL/SIGNED (SnnEncoderKind::Small) -- as opposed to
// EncoderDemo's generic textbook survey (TTFS/TTFS-log/Rate/Poisson).
//
//   TTFS:  a pure encoder -- number -> a single spike time, no neuron dynamics involved
//          (ttfsEncoder.cpp). Only defined for v in [0,1]; negative values simply never spike.
//   SMALL: NOT a spike-time formula -- a signed value becomes a conductance bump reinjected
//          into ONE of two neurons every tick (magnitude -> the negative-channel neuron if
//          v<0, the positive-channel neuron if v>=0; rlEncoder.cpp's encodeSmall, weight=1),
//          decayed with tau_exc=5ms between reapplications exactly like
//          snn_network.cpp's tickSmall()/propagateAndIntegrate(). Spikes only emerge if that
//          steady-state conductance is enough to cross the Izhikevich threshold.

namespace {
    constexpr int SCREEN_WIDTH = 1680;
    constexpr int SCREEN_HEIGHT = 900;

    // --- TTFS (ttfsEncoder.cpp, Mapping::LINEAR) ---
    constexpr double TTFS_DURATION_MS = 100.0;
    constexpr double TTFS_DT_MS = 1.0;
    constexpr double TTFS_THRESHOLD = 1e-9;
    constexpr double SWEEP_DURATION_MS = 1500.0; // wall-clock time for one cursor pass
    constexpr float FLASH_MAX_MS = 150.0f;

    std::vector<double> encodeTtfsLinear(double value) {
        if (value < 0.0 || value > 1.0 || value < TTFS_THRESHOLD) return {};
        double tMax = TTFS_DURATION_MS - TTFS_DT_MS;
        double t = (1.0 - value) * tMax;
        t = std::round(t / TTFS_DT_MS) * TTFS_DT_MS;
        return { std::clamp(t, 0.0, TTFS_DURATION_MS - TTFS_DT_MS) };
    }

    // --- SMALL / SIGNED (rlEncoder.cpp, RLEncoder::encodeSmall, weight=1.0) ---
    std::pair<double, double> encodeSmall(double value) {
        double magnitude = std::abs(value);
        return (value < 0.0) ? std::make_pair(magnitude, 0.0) : std::make_pair(0.0, magnitude);
    }

    constexpr double SIM_DT_MS = 0.5;
    constexpr double SMALL_SPEED_FACTOR = 0.1; // slow the Signed trace 10x so it's easy to follow
    constexpr double TAU_EXC = 5.0;
    constexpr double V_THRESHOLD = 30.0;
    constexpr double PLOT_V_MIN = -90.0;
    constexpr double PLOT_V_MAX = 40.0;
    constexpr double WINDOW_MS = 300.0;
    constexpr size_t HISTORY_CAPACITY = static_cast<size_t>(WINDOW_MS / SIM_DT_MS);

    struct SmallNeuron {
        SnnNeuronParams params;
        double v, u, gExc = 0.0;
        float flashMs = 0.0f;
        std::deque<float> history;

        SmallNeuron() : params(snnParamsFor(SnnNeuronType::RegularSpiking)), v(params.c), u(params.b * params.c) {
            history.assign(HISTORY_CAPACITY, static_cast<float>(v));
        }

        // Conductance bump reapplied every tick, exactly like tickSmall()/propagateAndIntegrate
        // in the main app's snn_network.cpp -- not a one-shot kick.
        void tick(double injectedCurrent) {
            gExc *= std::exp(-SIM_DT_MS / TAU_EXC);
            gExc += injectedCurrent;
            for (int sub = 0; sub < 2; ++sub) {
                double halfDt = SIM_DT_MS / 2.0;
                v += (0.04 * v * v + 5.0 * v + 140.0 - u + gExc) * halfDt;
                u += params.a * (params.b * v - u) * halfDt;
            }
            if (v >= V_THRESHOLD) {
                v = params.c;
                u += params.d;
                flashMs = FLASH_MAX_MS;
            }
            history.push_back(static_cast<float>(std::min(v, V_THRESHOLD)));
            if (history.size() > HISTORY_CAPACITY) history.pop_front();
        }
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

    void drawPanelFrame(Rectangle bounds, const std::string& title, const std::string& formula) {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(title.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), 20, DARKGRAY);
        DrawText(formula.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 32), 16, GRAY);
    }

    void drawTtfsPanel(Rectangle bounds, const std::vector<double>& spikes, const std::vector<float>& flashMs, double cursorTimeMs) {
        drawPanelFrame(bounds, "TTFS", "t = (1 - v) x t_max  (solo definido para v en [0,1])");

        Rectangle plot = { bounds.x + 20, bounds.y + 70, bounds.width - 40, bounds.height - 110 };
        float axisY = plot.y + plot.height * 0.5f;
        DrawLineEx({ plot.x, axisY }, { plot.x + plot.width, axisY }, 1.5f, LIGHTGRAY);
        DrawText("0 ms", static_cast<int>(plot.x), static_cast<int>(plot.y + plot.height + 6), 14, GRAY);
        std::string endLabel = TextFormat("%.0f ms", TTFS_DURATION_MS);
        int endLabelWidth = MeasureText(endLabel.c_str(), 14);
        DrawText(endLabel.c_str(), static_cast<int>(plot.x + plot.width) - endLabelWidth, static_cast<int>(plot.y + plot.height + 6), 14, GRAY);

        Color color = Color{60, 110, 220, 255};
        for (size_t i = 0; i < spikes.size(); ++i) {
            float x = plot.x + static_cast<float>(spikes[i] / TTFS_DURATION_MS) * plot.width;
            float flash = std::clamp(flashMs[i] / FLASH_MAX_MS, 0.0f, 1.0f);
            DrawLineEx({ x, axisY - plot.height * 0.35f }, { x, axisY + plot.height * 0.35f }, 3.0f, color);
            if (flash > 0.0f) DrawCircleV({ x, axisY }, 6.0f + 6.0f * flash, Fade(color, flash * 0.6f));
        }
        if (spikes.empty()) {
            const char* msg = "(sin spike: v esta bajo el umbral o es negativo)";
            float w = static_cast<float>(MeasureText(msg, 16));
            DrawText(msg, static_cast<int>(plot.x + plot.width / 2.0f - w / 2.0f), static_cast<int>(axisY - 10), 16, GRAY);
        }

        float cursorX = plot.x + static_cast<float>(cursorTimeMs / TTFS_DURATION_MS) * plot.width;
        DrawLineEx({ cursorX, plot.y - 6.0f }, { cursorX, plot.y + plot.height + 6.0f }, 2.0f, Fade(DARKGRAY, 0.6f));
    }

    void drawSmallPanel(Rectangle bounds, const std::string& title, const SmallNeuron& neuron, Color color) {
        drawPanelFrame(bounds, title, "corriente = |v| inyectada como conductancia cada tick (tau=5ms)");

        Rectangle plot = { bounds.x + 10, bounds.y + 60, bounds.width - 20, bounds.height - 70 };
        auto valueToY = [&](double v) {
            double t = (v - PLOT_V_MIN) / (PLOT_V_MAX - PLOT_V_MIN);
            return plot.y + plot.height - static_cast<float>(t) * plot.height;
        };

        float thresholdY = valueToY(V_THRESHOLD);
        for (float x = plot.x; x < plot.x + plot.width; x += 10.0f) {
            DrawLineEx({ x, thresholdY }, { std::min(x + 6.0f, plot.x + plot.width), thresholdY }, 1.0f, Fade(RED, 0.35f));
        }

        float xStep = plot.width / static_cast<float>(HISTORY_CAPACITY - 1);
        Vector2 prev{};
        for (size_t i = 0; i < neuron.history.size(); ++i) {
            Vector2 p = { plot.x + xStep * static_cast<float>(i), valueToY(neuron.history[i]) };
            if (i > 0) DrawLineEx(prev, p, 2.0f, color);
            prev = p;
        }
    }
}

int main() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Codificadores WANN: TTFS y Signed");
    SetTargetFPS(60);

    constexpr float MARGIN_X = 40.0f;
    constexpr float GRID_TOP = 30.0f;
    constexpr float GRID_BOTTOM = SCREEN_HEIGHT - 150.0f;
    constexpr float GAP = 20.0f;
    float colWidth = (SCREEN_WIDTH - 2.0f * MARGIN_X - GAP) / 2.0f;
    float smallRowHeight = (GRID_BOTTOM - GRID_TOP - GAP) / 2.0f;

    Rectangle ttfsBounds = { MARGIN_X, GRID_TOP, colWidth, GRID_BOTTOM - GRID_TOP };
    Rectangle smallNegBounds = { MARGIN_X + colWidth + GAP, GRID_TOP, colWidth, smallRowHeight };
    Rectangle smallPosBounds = { smallNegBounds.x, GRID_TOP + smallRowHeight + GAP, colWidth, smallRowHeight };

    Slider ttfsSlider{ { ttfsBounds.x, SCREEN_HEIGHT - 90.0f, colWidth, 10.0f }, 0.0f, 1.0f, 0.5f };
    Slider smallSlider{ { smallNegBounds.x, SCREEN_HEIGHT - 90.0f, colWidth, 10.0f }, -1.0f, 1.0f, 0.0f };

    SmallNeuron negNeuron, posNeuron;

    std::vector<double> ttfsSpikes = encodeTtfsLinear(ttfsSlider.value);
    std::vector<float> ttfsFlashMs(ttfsSpikes.size(), 0.0f);
    std::vector<bool> ttfsFired(ttfsSpikes.size(), false);
    double sweepProgress = 0.0;

    double accumulatorMs = 0.0;

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        ttfsSlider.update(mouse);
        smallSlider.update(mouse);

        float frameMs = static_cast<float>(GetFrameTime() * 1000.0);

        // TTFS: sweeping cursor, re-encoded (with the current slider value) every pass.
        sweepProgress += frameMs / SWEEP_DURATION_MS;
        if (sweepProgress >= 1.0) {
            sweepProgress -= 1.0;
            ttfsSpikes = encodeTtfsLinear(ttfsSlider.value);
            ttfsFlashMs.assign(ttfsSpikes.size(), 0.0f);
            ttfsFired.assign(ttfsSpikes.size(), false);
        }
        double cursorTimeMs = sweepProgress * TTFS_DURATION_MS;
        for (size_t i = 0; i < ttfsSpikes.size(); ++i) {
            if (!ttfsFired[i] && cursorTimeMs >= ttfsSpikes[i]) { ttfsFired[i] = true; ttfsFlashMs[i] = FLASH_MAX_MS; }
            ttfsFlashMs[i] = std::max(0.0f, ttfsFlashMs[i] - frameMs);
        }

        // Signed (SMALL): continuous Izhikevich simulation of the two channel neurons,
        // slowed down so individual spikes are easy to follow.
        auto [negCurrent, posCurrent] = encodeSmall(smallSlider.value);
        accumulatorMs += frameMs * SMALL_SPEED_FACTOR;
        accumulatorMs = std::min(accumulatorMs, SIM_DT_MS * 400.0);
        while (accumulatorMs >= SIM_DT_MS) {
            negNeuron.tick(negCurrent);
            posNeuron.tick(posCurrent);
            accumulatorMs -= SIM_DT_MS;
        }
        negNeuron.flashMs = std::max(0.0f, negNeuron.flashMs - frameMs);
        posNeuron.flashMs = std::max(0.0f, posNeuron.flashMs - frameMs);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        drawTtfsPanel(ttfsBounds, ttfsSpikes, ttfsFlashMs, cursorTimeMs);
        drawSmallPanel(smallNegBounds, "Signed - canal negativo", negNeuron, Color{220, 90, 70, 255});
        drawSmallPanel(smallPosBounds, "Signed - canal positivo", posNeuron, Color{60, 170, 90, 255});

        ttfsSlider.draw();
        DrawText(TextFormat("Valor TTFS: %.2f", ttfsSlider.value),
            static_cast<int>(ttfsSlider.track.x), static_cast<int>(ttfsSlider.track.y - 26), 20, DARKGRAY);

        smallSlider.draw();
        DrawText(TextFormat("Valor Signed: %.2f", smallSlider.value),
            static_cast<int>(smallSlider.track.x), static_cast<int>(smallSlider.track.y - 26), 20, DARKGRAY);

        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
