#pragma once
#include "demo_module.hpp"
#include "snn_network.hpp"

#include <deque>
#include <string>
#include <vector>

// Extracted from main_encoder_demo_wann.cpp: shows the two encoders actually used by the
// trained WANN models this project loads -- TTFS (SnnEncoderKind::Ttfs) and SMALL/SIGNED
// (SnnEncoderKind::Small) -- as opposed to EncoderModule's generic textbook survey. See the
// original file's header comment for the full explanation; unchanged here.
class EncoderWannModule : public IDemoModule {
public:
    EncoderWannModule();

    const char* name() const override { return "Codificadores WANN"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    struct SmallNeuron {
        SnnNeuronParams params;
        double v, u, gExc = 0.0;
        float flashMs = 0.0f;
        std::deque<float> history;

        SmallNeuron();
        void tick(double injectedCurrent);
    };

    struct Slider {
        Rectangle track{};
        float minValue = 0.0f, maxValue = 1.0f, value = 0.0f;
        bool dragging = false;

        void update(Vector2 mouse);
        void draw() const;
    };

    static void drawPanelFrame(Rectangle bounds, const std::string& title, const std::string& formula);
    void drawTtfsPanel(Rectangle bounds, const std::vector<double>& spikes, const std::vector<float>& flashMs, double cursorTimeMs) const;
    void drawSmallPanel(Rectangle bounds, const std::string& title, const SmallNeuron& neuron, Color color) const;

    Rectangle bounds_{};
    Rectangle ttfsBounds_{};
    Rectangle smallNegBounds_{};
    Rectangle smallPosBounds_{};

    Slider ttfsSlider_;
    Slider smallSlider_;
    SmallNeuron negNeuron_, posNeuron_;

    std::vector<double> ttfsSpikes_;
    std::vector<float> ttfsFlashMs_;
    std::vector<bool> ttfsFired_;
    double sweepProgress_ = 0.0;
    double cursorTimeMs_ = 0.0;
    double accumulatorMs_ = 0.0;
};
