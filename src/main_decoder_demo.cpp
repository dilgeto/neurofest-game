#include <raylib.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "../include/branding.hpp"
#include "../include/ui_scale.hpp"

// Standalone demo (separate binary from NeuroGame): shows how the two discrete decoders
// used by this project's WANN models (see rlDecoder.cpp's RLDecoder::decodeDiscreteAction)
// pick a winner among several output channels' spike trains -- and, critically, how they
// can disagree given the exact same spikes:
//   FIRST_SPIKE:  whichever channel's spike arrives earliest wins, decided instantly and
//                 locked in forever once that happens; every later spike is irrelevant.
//   RATE_ARGMAX:  whichever channel has the most spikes when the window ends wins (the
//                 "RATE_ARGMAX" name: argmax of spike_count/window*1000 -- since window is
//                 the same for every channel, that's equivalent to argmax of spike count).
//                 The leader can keep changing until the window closes.

namespace {
    // Right half of a 3840x2160 monitor.
    constexpr int SCREEN_WIDTH = 2160;
    constexpr int SCREEN_HEIGHT = 3840;
    constexpr int NUM_CHANNELS = 3;

    constexpr double WINDOW_MS = 100.0;
    constexpr double DT_MS = 1.0;
    constexpr double MAX_RATE = 100.0;           // Hz
    constexpr double SWEEP_DURATION_MS = 2000.0; // wall-clock time for one cursor pass
    constexpr float FLASH_MAX_MS = 150.0f;

    const Color CHANNEL_COLORS[NUM_CHANNELS] = {
        Color{60, 110, 220, 255},
        Color{230, 160, 60, 255},
        Color{60, 170, 90, 255},
    };

    // Same Bernoulli-per-tick formula as rateEncoder.cpp; stands in for "whatever spikes
    // the output layer produced" so channel activity is directly controllable per channel.
    std::vector<double> encodeRate(double value, std::mt19937& rng) {
        if (value < 0.0 || value > 1.0) return {};
        std::vector<double> spikes;
        double spikeProb = (value * MAX_RATE / 1000.0) * DT_MS;
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (double t = 0; t < WINDOW_MS; t += DT_MS)
            if (dis(rng) < spikeProb) spikes.push_back(t);
        return spikes;
    }

    struct Channel {
        std::string label;
        Color color;
        std::vector<double> spikes;
        std::vector<bool> fired;
        std::vector<float> flashMs;
        int countSoFar = 0;
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

        void draw(Color color) const {
            float trackY = track.y + track.height / 2.0f;
            DrawLineEx({ track.x, trackY }, { track.x + track.width, trackY }, 4.0f, LIGHTGRAY);
            float t = (value - minValue) / (maxValue - minValue);
            float handleX = track.x + t * track.width;
            DrawLineEx({ track.x, trackY }, { handleX, trackY }, 4.0f, color);
            DrawCircleV({ handleX, trackY }, 11.0f, color);
            DrawCircleLines(static_cast<int>(handleX), static_cast<int>(trackY), 11.0f, DARKGRAY);
        }
    };

    // Scales a base (1680-wide-reference) pixel size by g_uiScale, rounding to the nearest
    // integer -- use for every DrawText/MeasureText font-size argument.
    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }

    void drawChannelRow(Rectangle bounds, const Channel& channel, double cursorTimeMs) {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(channel.label.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), FS(18), DARKGRAY);

        Rectangle plot = { bounds.x + 20, bounds.y + 8, bounds.width - 40, bounds.height - 16 };
        float axisY = plot.y + plot.height * 0.5f;

        for (size_t i = 0; i < channel.spikes.size(); ++i) {
            float x = plot.x + static_cast<float>(channel.spikes[i] / WINDOW_MS) * plot.width;
            float flash = std::clamp(channel.flashMs[i] / FLASH_MAX_MS, 0.0f, 1.0f);
            DrawLineEx({ x, axisY - plot.height * 0.3f }, { x, axisY + plot.height * 0.3f }, 3.0f, channel.color);
            if (flash > 0.0f) DrawCircleV({ x, axisY }, 6.0f + 6.0f * flash, Fade(channel.color, flash * 0.6f));
        }

        float cursorX = plot.x + static_cast<float>(cursorTimeMs / WINDOW_MS) * plot.width;
        DrawLineEx({ cursorX, bounds.y + 2.0f }, { cursorX, bounds.y + bounds.height - 2.0f }, 2.0f, Fade(DARKGRAY, 0.6f));

        DrawText(TextFormat("%d", static_cast<int>(channel.spikes.size())),
            static_cast<int>(bounds.x + bounds.width - 30), static_cast<int>(bounds.y + 8), FS(18), DARKGRAY);
    }

    void drawVerdictPanel(Rectangle bounds, const std::string& title, const std::string& formula,
                           int winner, const std::vector<Channel>& channels, const std::string& detail) {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(title.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), FS(20), DARKGRAY);
        DrawText(formula.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 32), FS(15), GRAY);

        Vector2 badgeCenter = { bounds.x + bounds.width / 2.0f, bounds.y + bounds.height * 0.55f };
        if (winner < 0) {
            DrawCircleLines(static_cast<int>(badgeCenter.x), static_cast<int>(badgeCenter.y), 26.0f, GRAY);
            const char* msg = "Sin decidir";
            int w = MeasureText(msg, FS(18));
            DrawText(msg, static_cast<int>(badgeCenter.x - w / 2.0f), static_cast<int>(badgeCenter.y + 34), FS(18), GRAY);
        } else {
            DrawCircleV(badgeCenter, 26.0f, channels[static_cast<size_t>(winner)].color);
            DrawCircleLines(static_cast<int>(badgeCenter.x), static_cast<int>(badgeCenter.y), 26.0f, DARKGRAY);
            std::string msg = "Gana: " + channels[static_cast<size_t>(winner)].label;
            int w = MeasureText(msg.c_str(), FS(18));
            DrawText(msg.c_str(), static_cast<int>(badgeCenter.x - w / 2.0f), static_cast<int>(badgeCenter.y + 34), FS(18), DARKGRAY);
        }

        int detailWidth = MeasureText(detail.c_str(), FS(15));
        DrawText(detail.c_str(), static_cast<int>(bounds.x + bounds.width / 2.0f - detailWidth / 2.0f),
            static_cast<int>(bounds.y + bounds.height - 24), FS(15), GRAY);
    }
}

int main() {
    g_uiScale = SCREEN_WIDTH / 1680.0f;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Decodificadores: First Spike vs Rate/Argmax");
    SetWindowPosition(SCREEN_WIDTH, 0);
    SetTargetFPS(60);
    std::mt19937 rng(std::random_device{}());

    std::vector<Channel> channels(NUM_CHANNELS);
    for (int i = 0; i < NUM_CHANNELS; ++i) {
        channels[static_cast<size_t>(i)].label = TextFormat("Canal %d", i);
        channels[static_cast<size_t>(i)].color = CHANNEL_COLORS[i];
    }

    constexpr float MARGIN_X = 40.0f;
    constexpr float GRID_TOP = 30.0f;
    constexpr float GAP = 14.0f;
    constexpr float ROW_HEIGHT = 130.0f;
    constexpr float VERDICT_TOP = GRID_TOP + NUM_CHANNELS * (ROW_HEIGHT + GAP) + 10.0f;
    constexpr float VERDICT_HEIGHT = 190.0f;
    constexpr float SLIDER_Y = VERDICT_TOP + VERDICT_HEIGHT + 60.0f;

    float rowWidth = SCREEN_WIDTH - 2.0f * MARGIN_X;
    float verdictWidth = (rowWidth - GAP) / 2.0f;
    float sliderWidth = (rowWidth - 2.0f * GAP) / 3.0f;

    std::vector<Slider> sliders;
    for (int i = 0; i < NUM_CHANNELS; ++i) {
        sliders.push_back({ { MARGIN_X + i * (sliderWidth + GAP), SLIDER_Y, sliderWidth, 10.0f }, 0.0f, 1.0f, 0.4f });
    }

    int firstSpikeWinner = -1;
    double firstSpikeWinnerTimeMs = -1.0;
    int rateArgmaxWinner = -1;
    double sweepProgress = 0.0;

    auto regenerateAll = [&]() {
        for (int i = 0; i < NUM_CHANNELS; ++i) {
            Channel& c = channels[static_cast<size_t>(i)];
            c.spikes = encodeRate(sliders[static_cast<size_t>(i)].value, rng);
            c.fired.assign(c.spikes.size(), false);
            c.flashMs.assign(c.spikes.size(), 0.0f);
            c.countSoFar = 0;
        }
        firstSpikeWinner = -1;
        firstSpikeWinnerTimeMs = -1.0;
        rateArgmaxWinner = -1;
    };
    regenerateAll();

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        for (Slider& s : sliders) s.update(mouse);

        float frameMs = static_cast<float>(GetFrameTime() * 1000.0);
        sweepProgress += frameMs / SWEEP_DURATION_MS;
        if (sweepProgress >= 1.0) {
            sweepProgress -= 1.0;
            regenerateAll();
        }
        double cursorTimeMs = sweepProgress * WINDOW_MS;

        for (int i = 0; i < NUM_CHANNELS; ++i) {
            Channel& c = channels[static_cast<size_t>(i)];
            for (size_t j = 0; j < c.spikes.size(); ++j) {
                if (!c.fired[j] && cursorTimeMs >= c.spikes[j]) {
                    c.fired[j] = true;
                    c.flashMs[j] = FLASH_MAX_MS;
                    c.countSoFar++;
                    if (firstSpikeWinner == -1) { // globally first spike across all channels: cursor
                        firstSpikeWinner = i;      // reaches spikes in true chronological order, so
                        firstSpikeWinnerTimeMs = c.spikes[j]; // the first trigger IS the earliest one.
                    }
                    if (rateArgmaxWinner == -1 || c.countSoFar > channels[static_cast<size_t>(rateArgmaxWinner)].countSoFar) {
                        rateArgmaxWinner = i;
                    }
                }
                c.flashMs[j] = std::max(0.0f, c.flashMs[j] - frameMs);
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        for (int i = 0; i < NUM_CHANNELS; ++i) {
            Rectangle bounds = { MARGIN_X, GRID_TOP + i * (ROW_HEIGHT + GAP), rowWidth, ROW_HEIGHT };
            drawChannelRow(bounds, channels[static_cast<size_t>(i)], cursorTimeMs);
        }

        std::string firstSpikeDetail = (firstSpikeWinner < 0) ? "esperando la primera espiga..."
            : TextFormat("primera espiga a los %.0f ms", firstSpikeWinnerTimeMs);
        drawVerdictPanel(
            { MARGIN_X, VERDICT_TOP, verdictWidth, VERDICT_HEIGHT },
            "First Spike", "gana quien dispare primero (se decide una sola vez)",
            firstSpikeWinner, channels, firstSpikeDetail);

        std::string rateDetail = TextFormat("conteo: %d, %d, %d",
            channels[0].countSoFar, channels[1].countSoFar, channels[2].countSoFar);
        drawVerdictPanel(
            { MARGIN_X + verdictWidth + GAP, VERDICT_TOP, verdictWidth, VERDICT_HEIGHT },
            "Rate / Argmax", "gana quien tenga mas espigas (puede cambiar hasta el final)",
            rateArgmaxWinner, channels, rateDetail);

        for (int i = 0; i < NUM_CHANNELS; ++i) {
            sliders[static_cast<size_t>(i)].draw(CHANNEL_COLORS[i]);
            DrawText(TextFormat("Canal %d: %.2f", i, sliders[static_cast<size_t>(i)].value),
                static_cast<int>(sliders[static_cast<size_t>(i)].track.x),
                static_cast<int>(sliders[static_cast<size_t>(i)].track.y - 26), FS(18), DARKGRAY);
        }

        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
