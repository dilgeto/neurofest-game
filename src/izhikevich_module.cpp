#include "../include/izhikevich_module.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    constexpr double SIM_DT_MS = 0.5;       // Izhikevich integration step
    constexpr double WINDOW_MS = 300.0;     // how much history each trace shows
    constexpr size_t HISTORY_CAPACITY = static_cast<size_t>(WINDOW_MS / SIM_DT_MS);
    constexpr double V_THRESHOLD = 30.0;
    constexpr double PLOT_V_MIN = -90.0;
    constexpr double PLOT_V_MAX = 40.0;

    // Scales a base (1680-wide-reference) pixel size by g_uiScale, rounding to the nearest
    // integer -- use for every DrawText/MeasureText font-size argument.
    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }
}

IzhikevichModule::Neuron::Neuron(std::string label_, SnnNeuronType type)
    : label(std::move(label_)), params(snnParamsFor(type)), v(params.c), u(params.b * params.c) {
    history.assign(HISTORY_CAPACITY, static_cast<float>(v));
}

void IzhikevichModule::Neuron::step(double current) {
    for (int sub = 0; sub < 2; ++sub) {
        double halfDt = SIM_DT_MS / 2.0;
        v += (0.04 * v * v + 5.0 * v + 140.0 - u + current) * halfDt;
        u += params.a * (params.b * v - u) * halfDt;
    }
    if (v >= V_THRESHOLD) {
        v = params.c;
        u += params.d;
    }
    history.push_back(static_cast<float>(std::min(v, V_THRESHOLD)));
    if (history.size() > HISTORY_CAPACITY) history.pop_front();
}

// Restarts the neuron at rest under its *current* params -- used after dragging a/b/c/d to a
// combination whose trace has run off into an unhelpful state.
void IzhikevichModule::Neuron::reset() {
    v = params.c;
    u = params.b * params.c;
    history.assign(HISTORY_CAPACITY, static_cast<float>(v));
}

void IzhikevichModule::Slider::update(Vector2 mouse) {
    Rectangle hitArea = { track.x - 12, track.y - 12, track.width + 24, track.height + 24 };
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, hitArea)) dragging = true;
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragging = false;
    if (dragging) {
        float t = std::clamp((mouse.x - track.x) / track.width, 0.0f, 1.0f);
        value = minValue + t * (maxValue - minValue);
    }
}

void IzhikevichModule::Slider::draw() const {
    float trackY = track.y + track.height / 2.0f;
    DrawLineEx({ track.x, trackY }, { track.x + track.width, trackY }, 4.0f, LIGHTGRAY);
    float t = (value - minValue) / (maxValue - minValue);
    float handleX = track.x + t * track.width;
    DrawLineEx({ track.x, trackY }, { handleX, trackY }, 4.0f, Color{60, 110, 220, 255});
    DrawCircleV({ handleX, trackY }, 11.0f, Color{60, 110, 220, 255});
    DrawCircleLines(static_cast<int>(handleX), static_cast<int>(trackY), 11.0f, DARKGRAY);
}

void IzhikevichModule::drawNeuronPanel(Rectangle bounds, const Neuron& neuron) const {
    DrawRectangleRec(bounds, Color{250, 250, 251, 255});
    DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
    DrawText(neuron.label.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), FS(20), DARKGRAY);

    // Plot area, leaving room for the title.
    Rectangle plot = { bounds.x + 10, bounds.y + 36, bounds.width - 20, bounds.height - 46 };

    auto valueToY = [&](double v) {
        double t = (v - PLOT_V_MIN) / (PLOT_V_MAX - PLOT_V_MIN);
        return plot.y + plot.height - static_cast<float>(t) * plot.height;
    };

    // Threshold reference line.
    float thresholdY = valueToY(V_THRESHOLD);
    for (float x = plot.x; x < plot.x + plot.width; x += 10.0f) {
        DrawLineEx({ x, thresholdY }, { std::min(x + 6.0f, plot.x + plot.width), thresholdY }, 1.0f, Fade(RED, 0.35f));
    }

    // Voltage trace, oldest sample at the left edge.
    float xStep = plot.width / static_cast<float>(HISTORY_CAPACITY - 1);
    Vector2 prev{};
    for (size_t i = 0; i < neuron.history.size(); ++i) {
        Vector2 p = { plot.x + xStep * static_cast<float>(i), valueToY(neuron.history[i]) };
        if (i > 0) DrawLineEx(prev, p, 2.0f, Color{60, 110, 220, 255});
        prev = p;
    }
}

IzhikevichModule::IzhikevichModule()
    : visualizeButton_({0, 0, 1, 1}, "Visualizar patrones"),
      createButton_({0, 0, 1, 1}, "Crear patrón"),
      backButton_({0, 0, 1, 1}, "Volver"),
      playPauseButton_({0, 0, 1, 1}, "Iniciar"),
      resetButton_({0, 0, 1, 1}, "Reiniciar"),
      customNeuron_("Tu patrón", SnnNeuronType::RegularSpiking) {
    currentSlider_.minValue = 0.0f; currentSlider_.maxValue = 30.0f; currentSlider_.value = 10.0f;
    speedSlider_.minValue = 0.1f; speedSlider_.maxValue = 5.0f; speedSlider_.value = 0.1f;
    sliderA_.minValue = 0.01f; sliderA_.maxValue = 0.15f; sliderA_.value = 0.02f;
    sliderB_.minValue = 0.10f; sliderB_.maxValue = 0.30f; sliderB_.value = 0.20f;
    sliderC_.minValue = -75.0f; sliderC_.maxValue = -40.0f; sliderC_.value = -65.0f;
    sliderD_.minValue = 0.0f; sliderD_.maxValue = 10.0f; sliderD_.value = 8.0f;

    neurons_.emplace_back("Regular Spiking (RS)", SnnNeuronType::RegularSpiking);
    neurons_.emplace_back("Fast Spiking (FS)", SnnNeuronType::FastSpiking);
    neurons_.emplace_back("Chattering (CH)", SnnNeuronType::Chattering);
    neurons_.emplace_back("Low-Threshold Spiking (LTS)", SnnNeuronType::LowThresholdSpiking);
    neurons_.emplace_back("Intrinsically Bursting (IB)", SnnNeuronType::IntrinsicallyBursting);
    neurons_.emplace_back("Resonator (RZ)", SnnNeuronType::Resonator);
}

// Every position below is relative to bounds_ -- already correct as-is for however tall this
// panel ends up (a single grid row/control-bar-offset layout that fills whatever height it's
// given, unlike e.g. DecoderModule's fixed-budget layout): standalone at full window height,
// or a stacked share of it in the combined multi-demo view. Only the horizontal spacing is
// unconditionally safe to leave as absolute pixels, since every panel keeps the window's full
// width regardless of how many are stacked (see include/demo_module.hpp).
void IzhikevichModule::relayout() {
    const Rectangle& b = bounds_;

    constexpr float MENU_BUTTON_WIDTH = 340.0f;
    constexpr float MENU_BUTTON_HEIGHT = 60.0f;
    constexpr float MENU_GAP = 24.0f;
    visualizeButton_.setBounds({
        b.x + b.width / 2.0f - MENU_BUTTON_WIDTH / 2.0f, b.y + b.height / 2.0f - MENU_BUTTON_HEIGHT - MENU_GAP / 2.0f,
        MENU_BUTTON_WIDTH, MENU_BUTTON_HEIGHT
    });
    createButton_.setBounds({
        b.x + b.width / 2.0f - MENU_BUTTON_WIDTH / 2.0f, b.y + b.height / 2.0f + MENU_GAP / 2.0f,
        MENU_BUTTON_WIDTH, MENU_BUTTON_HEIGHT
    });
    backButton_.setBounds({ b.x + 20, b.y + b.height - 60.0f, 140, 40 });

    constexpr float BUTTON_WIDTH = 180.0f;
    constexpr float SLIDER_WIDTH = 300.0f;
    constexpr float CONTROL_GAP = 60.0f;
    float controlsTop = b.y + b.height - 140.0f;
    float controlsWidth = BUTTON_WIDTH + CONTROL_GAP + SLIDER_WIDTH + CONTROL_GAP + SLIDER_WIDTH;
    float controlsLeft = b.x + (b.width - controlsWidth) / 2.0f;

    playPauseButton_.setBounds({ controlsLeft, controlsTop, BUTTON_WIDTH, 50.0f });
    currentSlider_.track = { controlsLeft + BUTTON_WIDTH + CONTROL_GAP, controlsTop + 30.0f, SLIDER_WIDTH, 10.0f };
    speedSlider_.track = { controlsLeft + BUTTON_WIDTH + CONTROL_GAP + SLIDER_WIDTH + CONTROL_GAP, controlsTop + 30.0f, SLIDER_WIDTH, 10.0f };

    constexpr int COLS = 3;
    constexpr int ROWS = 2;
    constexpr float MARGIN_X = 40.0f;
    float gridTop = b.y + 30.0f;
    float gridBottom = controlsTop - 30.0f;
    constexpr float GAP = 20.0f;
    float panelWidth = (b.width - 2.0f * MARGIN_X - (COLS - 1) * GAP) / COLS;
    float panelHeight = (gridBottom - gridTop - (ROWS - 1) * GAP) / ROWS;

    for (int i = 0; i < static_cast<int>(neurons_.size()); ++i) {
        int row = i / COLS;
        int col = i % COLS;
        gridBounds_[i] = {
            b.x + MARGIN_X + col * (panelWidth + GAP),
            gridTop + row * (panelHeight + GAP),
            panelWidth, panelHeight
        };
    }

    float paramPanelLeft = b.x + MARGIN_X;
    constexpr float PARAM_SLIDER_WIDTH = 380.0f;
    constexpr float PARAM_ROW_H = 90.0f;
    float paramTop = gridTop + 60.0f;

    sliderA_.track = { paramPanelLeft, paramTop + 26.0f, PARAM_SLIDER_WIDTH, 10.0f };
    sliderB_.track = { paramPanelLeft, paramTop + PARAM_ROW_H + 26.0f, PARAM_SLIDER_WIDTH, 10.0f };
    sliderC_.track = { paramPanelLeft, paramTop + 2.0f * PARAM_ROW_H + 26.0f, PARAM_SLIDER_WIDTH, 10.0f };
    sliderD_.track = { paramPanelLeft, paramTop + 3.0f * PARAM_ROW_H + 26.0f, PARAM_SLIDER_WIDTH, 10.0f };
    resetButton_.setBounds({ paramPanelLeft, paramTop + 4.0f * PARAM_ROW_H, 180.0f, 44.0f });

    float plotLeft = paramPanelLeft + PARAM_SLIDER_WIDTH + 60.0f;
    plotBounds_ = { plotLeft, gridTop, b.x + b.width - MARGIN_X - plotLeft, gridBottom - gridTop };
}

void IzhikevichModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;
    relayout();
}

void IzhikevichModule::update(Vector2 mouse, float frameMs) {
    switch (screen_) {
        case DemoScreen::MENU:
            if (visualizeButton_.isClicked(mouse)) screen_ = DemoScreen::VISUALIZE;
            else if (createButton_.isClicked(mouse)) screen_ = DemoScreen::CREATE;
            break;

        case DemoScreen::VISUALIZE:
        case DemoScreen::CREATE: {
            if (backButton_.isClicked(mouse)) {
                screen_ = DemoScreen::MENU;
                break;
            }

            if (playPauseButton_.isClicked(mouse)) {
                running_ = !running_;
                playPauseButton_.setText(running_ ? "Pausar" : "Reanudar");
            }
            currentSlider_.update(mouse);
            speedSlider_.update(mouse);

            if (screen_ == DemoScreen::CREATE) {
                sliderA_.update(mouse);
                sliderB_.update(mouse);
                sliderC_.update(mouse);
                sliderD_.update(mouse);
                // Applied every frame so dragging any slider updates the live trace
                // immediately -- v/u themselves are left alone, only the equations'
                // coefficients change, exactly like turning a knob on a running circuit.
                customNeuron_.params.a = sliderA_.value;
                customNeuron_.params.b = sliderB_.value;
                customNeuron_.params.c = sliderC_.value;
                customNeuron_.params.d = sliderD_.value;
                if (resetButton_.isClicked(mouse)) customNeuron_.reset();
            }

            if (running_) {
                // "Velocidad" scales how much simulated time each real-world frame covers,
                // independent of the fixed SIM_DT_MS integration step.
                accumulatorMs_ += frameMs * speedSlider_.value;
                accumulatorMs_ = std::min(accumulatorMs_, SIM_DT_MS * 400.0); // cap catch-up on a stall
                while (accumulatorMs_ >= SIM_DT_MS) {
                    if (screen_ == DemoScreen::VISUALIZE) {
                        for (Neuron& neuron : neurons_) neuron.step(currentSlider_.value);
                    } else {
                        customNeuron_.step(currentSlider_.value);
                    }
                    accumulatorMs_ -= SIM_DT_MS;
                }
            }
            break;
        }
    }
}

void IzhikevichModule::draw(Vector2 mouse) const {
    switch (screen_) {
        case DemoScreen::MENU: {
            const char* title = "Patrones de Izhikevich";
            int titleWidth = MeasureText(title, FS(32));
            DrawText(title, static_cast<int>(bounds_.x + bounds_.width / 2.0f) - titleWidth / 2,
                static_cast<int>(bounds_.y + bounds_.height / 2.0f - 140.0f), FS(32), DARKGRAY);
            visualizeButton_.draw(mouse);
            createButton_.draw(mouse);
            break;
        }

        case DemoScreen::VISUALIZE: {
            for (int i = 0; i < static_cast<int>(neurons_.size()); ++i) {
                drawNeuronPanel(gridBounds_[i], neurons_[static_cast<size_t>(i)]);
            }

            backButton_.draw(mouse);
            playPauseButton_.draw(mouse);
            currentSlider_.draw();
            DrawText(TextFormat("Corriente: %.1f", currentSlider_.value),
                static_cast<int>(currentSlider_.track.x), static_cast<int>(currentSlider_.track.y - 26), FS(20), DARKGRAY);
            speedSlider_.draw();
            DrawText(TextFormat("Velocidad: %.2fx", speedSlider_.value),
                static_cast<int>(speedSlider_.track.x), static_cast<int>(speedSlider_.track.y - 26), FS(20), DARKGRAY);
            break;
        }

        case DemoScreen::CREATE: {
            DrawText("Crea tu patrón de disparo", static_cast<int>(bounds_.x + 40.0f), static_cast<int>(bounds_.y + 30.0f), FS(22), DARKGRAY);

            sliderA_.draw();
            DrawText(TextFormat("a (velocidad de recuperación de u): %.3f", sliderA_.value),
                static_cast<int>(sliderA_.track.x), static_cast<int>(sliderA_.track.y - 26), FS(18), DARKGRAY);
            sliderB_.draw();
            DrawText(TextFormat("b (sensibilidad de u a v): %.3f", sliderB_.value),
                static_cast<int>(sliderB_.track.x), static_cast<int>(sliderB_.track.y - 26), FS(18), DARKGRAY);
            sliderC_.draw();
            DrawText(TextFormat("c (voltaje de reinicio tras el pico): %.1f mV", sliderC_.value),
                static_cast<int>(sliderC_.track.x), static_cast<int>(sliderC_.track.y - 26), FS(18), DARKGRAY);
            sliderD_.draw();
            DrawText(TextFormat("d (incremento de u tras el pico): %.2f", sliderD_.value),
                static_cast<int>(sliderD_.track.x), static_cast<int>(sliderD_.track.y - 26), FS(18), DARKGRAY);
            resetButton_.draw(mouse);

            drawNeuronPanel(plotBounds_, customNeuron_);

            backButton_.draw(mouse);
            playPauseButton_.draw(mouse);
            currentSlider_.draw();
            DrawText(TextFormat("Corriente: %.1f", currentSlider_.value),
                static_cast<int>(currentSlider_.track.x), static_cast<int>(currentSlider_.track.y - 26), FS(20), DARKGRAY);
            speedSlider_.draw();
            DrawText(TextFormat("Velocidad: %.2fx", speedSlider_.value),
                static_cast<int>(speedSlider_.track.x), static_cast<int>(speedSlider_.track.y - 26), FS(20), DARKGRAY);
            break;
        }
    }
}
