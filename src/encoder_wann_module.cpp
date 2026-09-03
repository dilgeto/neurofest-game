#include "../include/encoder_wann_module.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
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

    // Scales a base (1680-wide-reference) pixel size by g_uiScale, rounding to the nearest
    // integer -- use for every DrawText/MeasureText font-size argument.
    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }
}

EncoderWannModule::SmallNeuron::SmallNeuron()
    : params(snnParamsFor(SnnNeuronType::RegularSpiking)), v(params.c), u(params.b * params.c) {
    history.assign(HISTORY_CAPACITY, static_cast<float>(v));
}

// Conductance bump reapplied every tick, exactly like tickSmall()/propagateAndIntegrate in
// the main app's snn_network.cpp -- not a one-shot kick.
void EncoderWannModule::SmallNeuron::tick(double injectedCurrent) {
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

void EncoderWannModule::Slider::update(Vector2 mouse) {
    Rectangle hitArea = { track.x - 12, track.y - 12, track.width + 24, track.height + 24 };
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, hitArea)) dragging = true;
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragging = false;
    if (dragging) {
        float t = std::clamp((mouse.x - track.x) / track.width, 0.0f, 1.0f);
        value = minValue + t * (maxValue - minValue);
    }
}

void EncoderWannModule::Slider::draw() const {
    float trackY = track.y + track.height / 2.0f;
    DrawLineEx({ track.x, trackY }, { track.x + track.width, trackY }, 4.0f, LIGHTGRAY);
    float t = (value - minValue) / (maxValue - minValue);
    float handleX = track.x + t * track.width;
    DrawLineEx({ track.x, trackY }, { handleX, trackY }, 4.0f, Color{60, 110, 220, 255});
    DrawCircleV({ handleX, trackY }, 11.0f, Color{60, 110, 220, 255});
    DrawCircleLines(static_cast<int>(handleX), static_cast<int>(trackY), 11.0f, DARKGRAY);
}

void EncoderWannModule::drawPanelFrame(Rectangle bounds, const std::string& title, const std::string& formula) {
    DrawRectangleRec(bounds, Color{250, 250, 251, 255});
    DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
    DrawText(title.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), FS(20), DARKGRAY);
    DrawText(formula.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 32), FS(16), GRAY);
}

void EncoderWannModule::drawTtfsPanel(Rectangle bounds, const std::vector<double>& spikes, const std::vector<float>& flashMs, double cursorTimeMs) const {
    drawPanelFrame(bounds, "TTFS", "t = (1 - v) x t_max  (solo definido para v en [0,1])");

    Rectangle plot = { bounds.x + 20, bounds.y + 70, bounds.width - 40, bounds.height - 110 };
    float axisY = plot.y + plot.height * 0.5f;
    DrawLineEx({ plot.x, axisY }, { plot.x + plot.width, axisY }, 1.5f, LIGHTGRAY);
    DrawText("0 ms", static_cast<int>(plot.x), static_cast<int>(plot.y + plot.height + 6), FS(14), GRAY);
    std::string endLabel = TextFormat("%.0f ms", TTFS_DURATION_MS);
    int endLabelWidth = MeasureText(endLabel.c_str(), FS(14));
    DrawText(endLabel.c_str(), static_cast<int>(plot.x + plot.width) - endLabelWidth, static_cast<int>(plot.y + plot.height + 6), FS(14), GRAY);

    Color color = Color{60, 110, 220, 255};
    for (size_t i = 0; i < spikes.size(); ++i) {
        float x = plot.x + static_cast<float>(spikes[i] / TTFS_DURATION_MS) * plot.width;
        float flash = std::clamp(flashMs[i] / FLASH_MAX_MS, 0.0f, 1.0f);
        DrawLineEx({ x, axisY - plot.height * 0.35f }, { x, axisY + plot.height * 0.35f }, 3.0f, color);
        if (flash > 0.0f) DrawCircleV({ x, axisY }, 6.0f + 6.0f * flash, Fade(color, flash * 0.6f));
    }
    if (spikes.empty()) {
        const char* msg = "(sin spike: v está bajo el umbral o es negativo)";
        float w = static_cast<float>(MeasureText(msg, FS(16)));
        DrawText(msg, static_cast<int>(plot.x + plot.width / 2.0f - w / 2.0f), static_cast<int>(axisY - 10), FS(16), GRAY);
    }

    float cursorX = plot.x + static_cast<float>(cursorTimeMs / TTFS_DURATION_MS) * plot.width;
    DrawLineEx({ cursorX, plot.y - 6.0f }, { cursorX, plot.y + plot.height + 6.0f }, 2.0f, Fade(DARKGRAY, 0.6f));
}

void EncoderWannModule::drawSmallPanel(Rectangle bounds, const std::string& title, const SmallNeuron& neuron, Color color) const {
    drawPanelFrame(bounds, title, "corriente = |v| inyectada como conductancia cada tick (tau=5ms)");

    Rectangle plot = { bounds.x + 10, bounds.y + 60, bounds.width - 20, bounds.height - 70 };
    auto valueToY = [&](double v) {
        double t = (v - PLOT_V_MIN) / (PLOT_V_MAX - PLOT_V_MIN);
        return plot.y + plot.height - static_cast<float>(t) * plot.height;
    };

    float thresholdY = static_cast<float>(valueToY(V_THRESHOLD));
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

EncoderWannModule::EncoderWannModule() {
    ttfsSlider_.minValue = 0.0f; ttfsSlider_.maxValue = 1.0f; ttfsSlider_.value = 0.5f;
    smallSlider_.minValue = -1.0f; smallSlider_.maxValue = 1.0f; smallSlider_.value = 0.0f;
    ttfsSpikes_ = encodeTtfsLinear(ttfsSlider_.value);
    ttfsFlashMs_.assign(ttfsSpikes_.size(), 0.0f);
    ttfsFired_.assign(ttfsSpikes_.size(), false);
}

// GRID_BOTTOM is already fully derived from bounds.height (fills whatever's available), so --
// like the other "fills available height" modules and unlike DecoderModule's fixed-pixel
// budget -- no fraction-of-a-reference-height treatment is needed here, just the bounds_
// offset/substitution.
void EncoderWannModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;
    const Rectangle& b = bounds;

    constexpr float MARGIN_X = 40.0f;
    constexpr float GRID_TOP = 30.0f;
    constexpr float GAP = 20.0f;
    float gridBottom = b.height - 150.0f;
    float colWidth = (b.width - 2.0f * MARGIN_X - GAP) / 2.0f;
    float smallRowHeight = (gridBottom - GRID_TOP - GAP) / 2.0f;

    ttfsBounds_ = { b.x + MARGIN_X, b.y + GRID_TOP, colWidth, gridBottom - GRID_TOP };
    smallNegBounds_ = { b.x + MARGIN_X + colWidth + GAP, b.y + GRID_TOP, colWidth, smallRowHeight };
    smallPosBounds_ = { smallNegBounds_.x, b.y + GRID_TOP + smallRowHeight + GAP, colWidth, smallRowHeight };

    ttfsSlider_.track = { ttfsBounds_.x, b.y + b.height - 90.0f, colWidth, 10.0f };
    smallSlider_.track = { smallNegBounds_.x, b.y + b.height - 90.0f, colWidth, 10.0f };
}

void EncoderWannModule::update(Vector2 mouse, float frameMs) {
    ttfsSlider_.update(mouse);
    smallSlider_.update(mouse);

    // TTFS: sweeping cursor, re-encoded (with the current slider value) every pass.
    sweepProgress_ += frameMs / SWEEP_DURATION_MS;
    if (sweepProgress_ >= 1.0) {
        sweepProgress_ -= 1.0;
        ttfsSpikes_ = encodeTtfsLinear(ttfsSlider_.value);
        ttfsFlashMs_.assign(ttfsSpikes_.size(), 0.0f);
        ttfsFired_.assign(ttfsSpikes_.size(), false);
    }
    cursorTimeMs_ = sweepProgress_ * TTFS_DURATION_MS;
    for (size_t i = 0; i < ttfsSpikes_.size(); ++i) {
        if (!ttfsFired_[i] && cursorTimeMs_ >= ttfsSpikes_[i]) { ttfsFired_[i] = true; ttfsFlashMs_[i] = FLASH_MAX_MS; }
        ttfsFlashMs_[i] = std::max(0.0f, ttfsFlashMs_[i] - frameMs);
    }

    // Signed (SMALL): continuous Izhikevich simulation of the two channel neurons, slowed
    // down so individual spikes are easy to follow.
    auto [negCurrent, posCurrent] = encodeSmall(smallSlider_.value);
    accumulatorMs_ += frameMs * SMALL_SPEED_FACTOR;
    accumulatorMs_ = std::min(accumulatorMs_, SIM_DT_MS * 400.0);
    while (accumulatorMs_ >= SIM_DT_MS) {
        negNeuron_.tick(negCurrent);
        posNeuron_.tick(posCurrent);
        accumulatorMs_ -= SIM_DT_MS;
    }
    negNeuron_.flashMs = std::max(0.0f, negNeuron_.flashMs - frameMs);
    posNeuron_.flashMs = std::max(0.0f, posNeuron_.flashMs - frameMs);
}

void EncoderWannModule::draw(Vector2 /*mouse*/) const {
    drawTtfsPanel(ttfsBounds_, ttfsSpikes_, ttfsFlashMs_, cursorTimeMs_);
    drawSmallPanel(smallNegBounds_, "Signed - canal negativo", negNeuron_, Color{220, 90, 70, 255});
    drawSmallPanel(smallPosBounds_, "Signed - canal positivo", posNeuron_, Color{60, 170, 90, 255});

    ttfsSlider_.draw();
    DrawText(TextFormat("Valor TTFS: %.2f", ttfsSlider_.value),
        static_cast<int>(ttfsSlider_.track.x), static_cast<int>(ttfsSlider_.track.y - 26), FS(20), DARKGRAY);

    smallSlider_.draw();
    DrawText(TextFormat("Valor Signed: %.2f", smallSlider_.value),
        static_cast<int>(smallSlider_.track.x), static_cast<int>(smallSlider_.track.y - 26), FS(20), DARKGRAY);
}
