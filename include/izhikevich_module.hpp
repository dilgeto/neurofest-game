#pragma once
#include "demo_module.hpp"
#include "button.hpp"
#include "snn_network.hpp"

#include <deque>
#include <string>
#include <vector>

// Extracted from main_izhikevich.cpp: starts on a small menu, then either (a) "Visualizar
// patrones" -- animates the 6 canonical Izhikevich neuron behaviors side by side under a
// shared, user-controlled DC current injection; or (b) "Crear patron" -- lets the user drag
// a/b/c/d sliders themselves and watch a single neuron's trace update live. No synapses/
// network here: each neuron is simulated independently. See the original file's header
// comment for the full explanation; unchanged here.
class IzhikevichModule : public IDemoModule {
public:
    IzhikevichModule();

    const char* name() const override { return "Patrones de Izhikevich"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    enum class DemoScreen { MENU, VISUALIZE, CREATE };

    struct Neuron {
        std::string label;
        SnnNeuronParams params;
        double v;
        double u;
        std::deque<float> history;

        Neuron(std::string label_, SnnNeuronType type);
        void step(double current);
        void reset();
    };

    struct Slider {
        Rectangle track{};
        float minValue = 0.0f, maxValue = 1.0f, value = 0.0f;
        bool dragging = false;

        void update(Vector2 mouse);
        void draw() const;
    };

    void drawNeuronPanel(Rectangle bounds, const Neuron& neuron) const;
    void relayout(); // recomputes every button/slider/panel position from bounds_

    Rectangle bounds_{};
    DemoScreen screen_ = DemoScreen::MENU;

    Button visualizeButton_;
    Button createButton_;
    Button backButton_;
    Button playPauseButton_;
    Button resetButton_;

    Slider currentSlider_;
    Slider speedSlider_;
    Slider sliderA_, sliderB_, sliderC_, sliderD_;

    std::vector<Neuron> neurons_;
    Neuron customNeuron_;

    // Panel rectangles recomputed by relayout(), reused by both update (hit testing isn't
    // needed here, just draw) and draw.
    Rectangle gridBounds_[6]{};
    Rectangle plotBounds_{};

    bool running_ = false;
    double accumulatorMs_ = 0.0;
};
