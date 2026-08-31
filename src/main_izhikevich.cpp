#include <raylib.h>
#include <algorithm>
#include <deque>
#include <string>
#include <vector>

#include "../include/branding.hpp"
#include "../include/button.hpp"
#include "../include/snn_network.hpp"

// Standalone demo (separate binary from NeuroGame): starts on a small menu, then either
// (a) "Visualizar patrones" -- animates the 6 canonical Izhikevich neuron behaviors side by
// side under a shared, user-controlled DC current injection, the same a/b/c/d parameter
// table used by the SNN engine (snn_network.hpp's snnParamsFor()); or (b) "Crear patron" --
// lets the user drag a/b/c/d sliders themselves and watch a single neuron's trace update
// live. No synapses/network here: each neuron is simulated independently as
// `v' = 0.04v^2+5v+140-u+I` (matches snn-simulator/src/core/neuron.cpp).

namespace {
    constexpr int SCREEN_WIDTH = 1680;
    constexpr int SCREEN_HEIGHT = 900;

    constexpr double SIM_DT_MS = 0.5;       // Izhikevich integration step
    constexpr double WINDOW_MS = 300.0;     // how much history each trace shows
    constexpr size_t HISTORY_CAPACITY = static_cast<size_t>(WINDOW_MS / SIM_DT_MS);
    constexpr double V_THRESHOLD = 30.0;
    constexpr double PLOT_V_MIN = -90.0;
    constexpr double PLOT_V_MAX = 40.0;

    enum class DemoScreen { MENU, VISUALIZE, CREATE };

    struct IzhikevichNeuron {
        std::string label;
        SnnNeuronParams params;
        double v;
        double u;
        std::deque<float> history;

        explicit IzhikevichNeuron(std::string label_, SnnNeuronType type)
            : label(std::move(label_)), params(snnParamsFor(type)), v(params.c), u(params.b * params.c) {
            history.assign(HISTORY_CAPACITY, static_cast<float>(v));
        }

        void step(double current) {
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

        // Restarts the neuron at rest under its *current* params -- used after dragging
        // a/b/c/d to a combination whose trace has run off into an unhelpful state.
        void reset() {
            v = params.c;
            u = params.b * params.c;
            history.assign(HISTORY_CAPACITY, static_cast<float>(v));
        }
    };

    struct Slider {
        Rectangle track;
        float minValue;
        float maxValue;
        float value;
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

    void drawNeuronPanel(Rectangle bounds, const IzhikevichNeuron& neuron) {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(neuron.label.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), 20, DARKGRAY);

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
}

int main() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Comportamientos de Izhikevich");
    SetTargetFPS(60);

    DemoScreen screen = DemoScreen::MENU;

    // Menu: two stacked buttons choosing between the two screens below.
    constexpr float MENU_BUTTON_WIDTH = 340.0f;
    constexpr float MENU_BUTTON_HEIGHT = 60.0f;
    constexpr float MENU_GAP = 24.0f;
    Button visualizeButton({
        SCREEN_WIDTH / 2.0f - MENU_BUTTON_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f - MENU_BUTTON_HEIGHT - MENU_GAP / 2.0f,
        MENU_BUTTON_WIDTH, MENU_BUTTON_HEIGHT
    }, "Visualizar patrones");
    Button createButton({
        SCREEN_WIDTH / 2.0f - MENU_BUTTON_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f + MENU_GAP / 2.0f,
        MENU_BUTTON_WIDTH, MENU_BUTTON_HEIGHT
    }, "Crear patron");
    Button backButton({ 20, SCREEN_HEIGHT - 60.0f, 140, 40 }, "Volver");

    // Bottom, horizontally-centered control bar shared by both screens:
    // [Play/Pause] [Corriente slider] [Velocidad slider].
    constexpr float BUTTON_WIDTH = 180.0f;
    constexpr float SLIDER_WIDTH = 300.0f;
    constexpr float CONTROL_GAP = 60.0f;
    constexpr float CONTROLS_TOP = SCREEN_HEIGHT - 140.0f;
    constexpr float CONTROLS_WIDTH = BUTTON_WIDTH + CONTROL_GAP + SLIDER_WIDTH + CONTROL_GAP + SLIDER_WIDTH;
    constexpr float CONTROLS_LEFT = (SCREEN_WIDTH - CONTROLS_WIDTH) / 2.0f;

    Button playPauseButton({ CONTROLS_LEFT, CONTROLS_TOP, BUTTON_WIDTH, 50.0f }, "Iniciar");
    Slider currentSlider{
        { CONTROLS_LEFT + BUTTON_WIDTH + CONTROL_GAP, CONTROLS_TOP + 30.0f, SLIDER_WIDTH, 10.0f },
        0.0f, 30.0f, 10.0f
    };
    Slider speedSlider{
        { CONTROLS_LEFT + BUTTON_WIDTH + CONTROL_GAP + SLIDER_WIDTH + CONTROL_GAP, CONTROLS_TOP + 30.0f, SLIDER_WIDTH, 10.0f },
        0.1f, 5.0f, 0.1f
    };

    // "Visualizar patrones": the original 6-panel gallery.
    std::vector<IzhikevichNeuron> neurons;
    neurons.emplace_back("Regular Spiking (RS)", SnnNeuronType::RegularSpiking);
    neurons.emplace_back("Fast Spiking (FS)", SnnNeuronType::FastSpiking);
    neurons.emplace_back("Chattering (CH)", SnnNeuronType::Chattering);
    neurons.emplace_back("Low-Threshold Spiking (LTS)", SnnNeuronType::LowThresholdSpiking);
    neurons.emplace_back("Intrinsically Bursting (IB)", SnnNeuronType::IntrinsicallyBursting);
    neurons.emplace_back("Resonator (RZ)", SnnNeuronType::Resonator);

    constexpr int COLS = 3;
    constexpr int ROWS = 2;
    constexpr float MARGIN_X = 40.0f;
    constexpr float GRID_TOP = 30.0f;
    constexpr float GRID_BOTTOM = CONTROLS_TOP - 30.0f;
    constexpr float GAP = 20.0f;
    float panelWidth = (SCREEN_WIDTH - 2.0f * MARGIN_X - (COLS - 1) * GAP) / COLS;
    float panelHeight = (GRID_BOTTOM - GRID_TOP - (ROWS - 1) * GAP) / ROWS;

    // "Crear patron": a/b/c/d sliders (ranged around Izhikevich's canonical RS/FS/CH/LTS/
    // IB/RZ table above, with margin for combinations outside those 6 presets) driving one
    // live neuron, plotted big on the right.
    constexpr float PARAM_PANEL_LEFT = MARGIN_X;
    constexpr float PARAM_SLIDER_WIDTH = 380.0f;
    constexpr float PARAM_ROW_H = 90.0f;
    constexpr float PARAM_TOP = GRID_TOP + 60.0f;

    Slider sliderA{ { PARAM_PANEL_LEFT, PARAM_TOP + 26.0f, PARAM_SLIDER_WIDTH, 10.0f }, 0.01f, 0.15f, 0.02f };
    Slider sliderB{ { PARAM_PANEL_LEFT, PARAM_TOP + PARAM_ROW_H + 26.0f, PARAM_SLIDER_WIDTH, 10.0f }, 0.10f, 0.30f, 0.20f };
    Slider sliderC{ { PARAM_PANEL_LEFT, PARAM_TOP + 2.0f * PARAM_ROW_H + 26.0f, PARAM_SLIDER_WIDTH, 10.0f }, -75.0f, -40.0f, -65.0f };
    Slider sliderD{ { PARAM_PANEL_LEFT, PARAM_TOP + 3.0f * PARAM_ROW_H + 26.0f, PARAM_SLIDER_WIDTH, 10.0f }, 0.0f, 10.0f, 8.0f };
    Button resetButton({ PARAM_PANEL_LEFT, PARAM_TOP + 4.0f * PARAM_ROW_H, 180.0f, 44.0f }, "Reiniciar");

    IzhikevichNeuron customNeuron("Tu patron", SnnNeuronType::RegularSpiking);

    float plotLeft = PARAM_PANEL_LEFT + PARAM_SLIDER_WIDTH + 60.0f;
    Rectangle plotBounds = { plotLeft, GRID_TOP, SCREEN_WIDTH - MARGIN_X - plotLeft, GRID_BOTTOM - GRID_TOP };

    bool running = false;
    double accumulatorMs = 0.0;

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();

        switch (screen) {
            case DemoScreen::MENU:
                if (visualizeButton.isClicked(mouse)) screen = DemoScreen::VISUALIZE;
                else if (createButton.isClicked(mouse)) screen = DemoScreen::CREATE;
                break;

            case DemoScreen::VISUALIZE:
            case DemoScreen::CREATE: {
                if (backButton.isClicked(mouse)) {
                    screen = DemoScreen::MENU;
                    break;
                }

                if (playPauseButton.isClicked(mouse)) {
                    running = !running;
                    playPauseButton.setText(running ? "Pausar" : "Reanudar");
                }
                currentSlider.update(mouse);
                speedSlider.update(mouse);

                if (screen == DemoScreen::CREATE) {
                    sliderA.update(mouse);
                    sliderB.update(mouse);
                    sliderC.update(mouse);
                    sliderD.update(mouse);
                    // Applied every frame so dragging any slider updates the live trace
                    // immediately -- v/u themselves are left alone, only the equations'
                    // coefficients change, exactly like turning a knob on a running circuit.
                    customNeuron.params.a = sliderA.value;
                    customNeuron.params.b = sliderB.value;
                    customNeuron.params.c = sliderC.value;
                    customNeuron.params.d = sliderD.value;
                    if (resetButton.isClicked(mouse)) customNeuron.reset();
                }

                if (running) {
                    // "Velocidad" scales how much simulated time each real-world frame
                    // covers, independent of the fixed SIM_DT_MS integration step (so the
                    // Izhikevich math itself doesn't change, only how fast we play it back).
                    accumulatorMs += GetFrameTime() * 1000.0 * speedSlider.value;
                    accumulatorMs = std::min(accumulatorMs, SIM_DT_MS * 400.0); // cap catch-up on a stall
                    while (accumulatorMs >= SIM_DT_MS) {
                        if (screen == DemoScreen::VISUALIZE) {
                            for (IzhikevichNeuron& neuron : neurons) neuron.step(currentSlider.value);
                        } else {
                            customNeuron.step(currentSlider.value);
                        }
                        accumulatorMs -= SIM_DT_MS;
                    }
                }
                break;
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        switch (screen) {
            case DemoScreen::MENU: {
                const char* title = "Patrones de Izhikevich";
                int titleWidth = MeasureText(title, 32);
                DrawText(title, SCREEN_WIDTH / 2 - titleWidth / 2, static_cast<int>(SCREEN_HEIGHT / 2.0f - 140.0f), 32, DARKGRAY);
                visualizeButton.draw(mouse);
                createButton.draw(mouse);
                break;
            }

            case DemoScreen::VISUALIZE: {
                for (int i = 0; i < static_cast<int>(neurons.size()); ++i) {
                    int row = i / COLS;
                    int col = i % COLS;
                    Rectangle bounds = {
                        MARGIN_X + col * (panelWidth + GAP),
                        GRID_TOP + row * (panelHeight + GAP),
                        panelWidth, panelHeight
                    };
                    drawNeuronPanel(bounds, neurons[static_cast<size_t>(i)]);
                }

                backButton.draw(mouse);
                playPauseButton.draw(mouse);
                currentSlider.draw();
                DrawText(TextFormat("Corriente: %.1f", currentSlider.value),
                    static_cast<int>(currentSlider.track.x), static_cast<int>(currentSlider.track.y - 26), 20, DARKGRAY);
                speedSlider.draw();
                DrawText(TextFormat("Velocidad: %.2fx", speedSlider.value),
                    static_cast<int>(speedSlider.track.x), static_cast<int>(speedSlider.track.y - 26), 20, DARKGRAY);
                break;
            }

            case DemoScreen::CREATE: {
                DrawText("Crea tu patron de disparo", static_cast<int>(PARAM_PANEL_LEFT), static_cast<int>(GRID_TOP), 22, DARKGRAY);

                sliderA.draw();
                DrawText(TextFormat("a (velocidad de recuperacion de u): %.3f", sliderA.value),
                    static_cast<int>(sliderA.track.x), static_cast<int>(sliderA.track.y - 26), 18, DARKGRAY);
                sliderB.draw();
                DrawText(TextFormat("b (sensibilidad de u a v): %.3f", sliderB.value),
                    static_cast<int>(sliderB.track.x), static_cast<int>(sliderB.track.y - 26), 18, DARKGRAY);
                sliderC.draw();
                DrawText(TextFormat("c (voltaje de reinicio tras el pico): %.1f mV", sliderC.value),
                    static_cast<int>(sliderC.track.x), static_cast<int>(sliderC.track.y - 26), 18, DARKGRAY);
                sliderD.draw();
                DrawText(TextFormat("d (incremento de u tras el pico): %.2f", sliderD.value),
                    static_cast<int>(sliderD.track.x), static_cast<int>(sliderD.track.y - 26), 18, DARKGRAY);
                resetButton.draw(mouse);

                drawNeuronPanel(plotBounds, customNeuron);

                backButton.draw(mouse);
                playPauseButton.draw(mouse);
                currentSlider.draw();
                DrawText(TextFormat("Corriente: %.1f", currentSlider.value),
                    static_cast<int>(currentSlider.track.x), static_cast<int>(currentSlider.track.y - 26), 20, DARKGRAY);
                speedSlider.draw();
                DrawText(TextFormat("Velocidad: %.2fx", speedSlider.value),
                    static_cast<int>(speedSlider.track.x), static_cast<int>(speedSlider.track.y - 26), 20, DARKGRAY);
                break;
            }
        }

        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
