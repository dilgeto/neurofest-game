#pragma once
#include "demo_module.hpp"

#include <random>
#include <string>
#include <vector>

// Extracted from main_encoder_demo.cpp: shows how 2 of snn-simulator's spike encoders turn a
// single number in [0,1] into a spike train -- one temporal (TTFS linear) encoder, one rate
// encoder. See the original file's header comment for the full explanation; unchanged here.
class EncoderModule : public IDemoModule {
public:
    EncoderModule();

    const char* name() const override { return "Codificadores"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    struct Panel {
        std::string title;
        std::string formula;
        Color color{};
        std::vector<double> spikes;
        std::vector<bool> fired;     // has the sweep cursor already crossed this spike?
        std::vector<float> flashMs;  // brief highlight pulse when the cursor crosses it
    };

    struct Slider {
        Rectangle track{};
        float minValue = 0.0f, maxValue = 1.0f, value = 0.5f;
        bool dragging = false;

        void update(Vector2 mouse);
        void draw() const;
    };

    void regenerateAll();
    static void drawPanel(Rectangle bounds, const Panel& panel, double cursorTimeMs);

    Rectangle bounds_{};
    Rectangle panelBounds_[2]{};

    std::mt19937 rng_;
    std::vector<Panel> panels_;
    Slider valueSlider_;
    double sweepProgress_ = 0.0;
    double cursorTimeMs_ = 0.0;
};
