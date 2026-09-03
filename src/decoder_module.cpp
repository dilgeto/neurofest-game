#include "../include/decoder_module.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    constexpr double WINDOW_MS = 100.0;
    constexpr double DT_MS = 1.0;
    constexpr double MAX_RATE = 100.0;           // Hz
    constexpr double SWEEP_DURATION_MS = 2000.0; // wall-clock time for one cursor pass
    constexpr float FLASH_MAX_MS = 150.0f;

    const Color CHANNEL_COLORS[DecoderModule::NUM_CHANNELS] = {
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

    // Scales a base (1680-wide-reference) pixel size by g_uiScale, rounding to the nearest
    // integer -- use for every DrawText/MeasureText font-size argument. Width-based (panel
    // width never shrinks in the stacked multi-demo layout), so unlike the vertical layout
    // math in setBounds() this needs no bounds.height-relative treatment.
    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }
}

DecoderModule::DecoderModule() : rng_(std::random_device{}()), channels_(NUM_CHANNELS), sliders_(NUM_CHANNELS) {
    for (int i = 0; i < NUM_CHANNELS; ++i) {
        channels_[static_cast<size_t>(i)].label = TextFormat("Canal %d", i);
        channels_[static_cast<size_t>(i)].color = CHANNEL_COLORS[i];
    }
    regenerateAll();
}

void DecoderModule::regenerateAll() {
    for (int i = 0; i < NUM_CHANNELS; ++i) {
        Channel& c = channels_[static_cast<size_t>(i)];
        c.spikes = encodeRate(sliders_[static_cast<size_t>(i)].value, rng_);
        c.fired.assign(c.spikes.size(), false);
        c.flashMs.assign(c.spikes.size(), 0.0f);
        c.countSoFar = 0;
    }
    firstSpikeWinner_ = -1;
    firstSpikeWinnerTimeMs_ = -1.0;
    rateArgmaxWinner_ = -1;
}

void DecoderModule::Slider::update(Vector2 mouse) {
    Rectangle hitArea = { track.x - 12, track.y - 12, track.width + 24, track.height + 24 };
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, hitArea)) dragging = true;
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragging = false;
    if (dragging) {
        float t = std::clamp((mouse.x - track.x) / track.width, 0.0f, 1.0f);
        value = minValue + t * (maxValue - minValue);
    }
}

void DecoderModule::Slider::draw(Color color) const {
    float trackY = track.y + track.height / 2.0f;
    DrawLineEx({ track.x, trackY }, { track.x + track.width, trackY }, 4.0f, LIGHTGRAY);
    float t = (value - minValue) / (maxValue - minValue);
    float handleX = track.x + t * track.width;
    DrawLineEx({ track.x, trackY }, { handleX, trackY }, 4.0f, color);
    DrawCircleV({ handleX, trackY }, 11.0f, color);
    DrawCircleLines(static_cast<int>(handleX), static_cast<int>(trackY), 11.0f, DARKGRAY);
}

void DecoderModule::drawChannelRow(Rectangle bounds, const Channel& channel, double cursorTimeMs) {
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

void DecoderModule::drawVerdictPanel(Rectangle bounds, const std::string& title, const std::string& formula,
                                      int winner, const std::string& detail) const {
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
        DrawCircleV(badgeCenter, 26.0f, channels_[static_cast<size_t>(winner)].color);
        DrawCircleLines(static_cast<int>(badgeCenter.x), static_cast<int>(badgeCenter.y), 26.0f, DARKGRAY);
        std::string msg = "Gana: " + channels_[static_cast<size_t>(winner)].label;
        int w = MeasureText(msg.c_str(), FS(18));
        DrawText(msg.c_str(), static_cast<int>(badgeCenter.x - w / 2.0f), static_cast<int>(badgeCenter.y + 34), FS(18), DARKGRAY);
    }

    int detailWidth = MeasureText(detail.c_str(), FS(15));
    DrawText(detail.c_str(), static_cast<int>(bounds.x + bounds.width / 2.0f - detailWidth / 2.0f),
        static_cast<int>(bounds.y + bounds.height - 24), FS(15), GRAY);
}

// Vertical layout below is expressed as a fraction of bounds.height, relative to REF_H -- the
// ~900px-tall canvas these pixel values (ROW_HEIGHT=130 etc.) were originally tuned against --
// so the whole block scales to fill whatever height this module is given: the full window
// standalone, or a stacked share of it in the combined multi-demo view. Horizontal spacing
// doesn't need this treatment: every panel keeps the window's full width regardless of how
// many are stacked (see include/demo_module.hpp).
void DecoderModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;

    constexpr float REF_H = 900.0f;
    auto vf = [&](float basePx) { return basePx / REF_H * bounds.height; };

    float gridTop = bounds.y + vf(30.0f);
    float gapY = vf(14.0f);
    float rowHeight = vf(130.0f);
    float verdictTop = gridTop + NUM_CHANNELS * (rowHeight + gapY) + vf(10.0f);
    float verdictHeight = vf(190.0f);
    float sliderY = verdictTop + verdictHeight + vf(60.0f);

    constexpr float MARGIN_X = 40.0f;
    constexpr float GAP_X = 14.0f;
    float rowWidth = bounds.width - 2.0f * MARGIN_X;
    float verdictWidth = (rowWidth - GAP_X) / 2.0f;
    float sliderWidth = (rowWidth - 2.0f * GAP_X) / 3.0f;

    for (int i = 0; i < NUM_CHANNELS; ++i) {
        channelBounds_[i] = { bounds.x + MARGIN_X, gridTop + i * (rowHeight + gapY), rowWidth, rowHeight };
    }
    verdictBounds_[0] = { bounds.x + MARGIN_X, verdictTop, verdictWidth, verdictHeight };
    verdictBounds_[1] = { bounds.x + MARGIN_X + verdictWidth + GAP_X, verdictTop, verdictWidth, verdictHeight };

    for (int i = 0; i < NUM_CHANNELS; ++i) {
        sliders_[static_cast<size_t>(i)].track = { bounds.x + MARGIN_X + i * (sliderWidth + GAP_X), sliderY, sliderWidth, 10.0f };
    }
}

void DecoderModule::update(Vector2 mouse, float frameMs) {
    for (Slider& s : sliders_) s.update(mouse);

    sweepProgress_ += frameMs / SWEEP_DURATION_MS;
    if (sweepProgress_ >= 1.0) {
        sweepProgress_ -= 1.0;
        regenerateAll();
    }
    cursorTimeMs_ = sweepProgress_ * WINDOW_MS;

    for (int i = 0; i < NUM_CHANNELS; ++i) {
        Channel& c = channels_[static_cast<size_t>(i)];
        for (size_t j = 0; j < c.spikes.size(); ++j) {
            if (!c.fired[j] && cursorTimeMs_ >= c.spikes[j]) {
                c.fired[j] = true;
                c.flashMs[j] = FLASH_MAX_MS;
                c.countSoFar++;
                if (firstSpikeWinner_ == -1) { // globally first spike across all channels: cursor
                    firstSpikeWinner_ = i;      // reaches spikes in true chronological order, so
                    firstSpikeWinnerTimeMs_ = c.spikes[j]; // the first trigger IS the earliest one.
                }
                if (rateArgmaxWinner_ == -1 || c.countSoFar > channels_[static_cast<size_t>(rateArgmaxWinner_)].countSoFar) {
                    rateArgmaxWinner_ = i;
                }
            }
            c.flashMs[j] = std::max(0.0f, c.flashMs[j] - frameMs);
        }
    }
}

void DecoderModule::draw(Vector2 /*mouse*/) const {
    for (int i = 0; i < NUM_CHANNELS; ++i) {
        drawChannelRow(channelBounds_[i], channels_[static_cast<size_t>(i)], cursorTimeMs_);
    }

    std::string firstSpikeDetail = (firstSpikeWinner_ < 0) ? "esperando la primera espiga..."
        : TextFormat("primera espiga a los %.0f ms", firstSpikeWinnerTimeMs_);
    drawVerdictPanel(verdictBounds_[0], "First Spike", "gana quien dispare primero (se decide una sola vez)",
        firstSpikeWinner_, firstSpikeDetail);

    std::string rateDetail = TextFormat("conteo: %d, %d, %d",
        channels_[0].countSoFar, channels_[1].countSoFar, channels_[2].countSoFar);
    drawVerdictPanel(verdictBounds_[1], "Rate / Argmax", "gana quien tenga más espigas (puede cambiar hasta el final)",
        rateArgmaxWinner_, rateDetail);

    for (int i = 0; i < NUM_CHANNELS; ++i) {
        sliders_[static_cast<size_t>(i)].draw(CHANNEL_COLORS[i]);
        DrawText(TextFormat("Canal %d: %.2f", i, sliders_[static_cast<size_t>(i)].value),
            static_cast<int>(sliders_[static_cast<size_t>(i)].track.x),
            static_cast<int>(sliders_[static_cast<size_t>(i)].track.y - 26), FS(18), DARKGRAY);
    }
}
