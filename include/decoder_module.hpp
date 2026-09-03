#pragma once
#include "demo_module.hpp"

#include <random>
#include <string>
#include <vector>

// Extracted from main_decoder_demo.cpp: shows how the two discrete decoders used by this
// project's WANN models (see rlDecoder.cpp's RLDecoder::decodeDiscreteAction) pick a winner
// among several output channels' spike trains, and how they can disagree given the exact
// same spikes -- FIRST_SPIKE (earliest spike wins, decided once) vs RATE_ARGMAX (most spikes
// when the window ends wins, can flip until then). See main_decoder_demo.cpp's original
// header comment for the full explanation; unchanged here.
class DecoderModule : public IDemoModule {
public:
    static constexpr int NUM_CHANNELS = 3;

    DecoderModule();

    const char* name() const override { return "Decodificadores"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    struct Channel {
        std::string label;
        Color color{};
        std::vector<double> spikes;
        std::vector<bool> fired;
        std::vector<float> flashMs;
        int countSoFar = 0;
    };

    struct Slider {
        Rectangle track{};
        float minValue = 0.0f, maxValue = 1.0f, value = 0.4f;
        bool dragging = false;

        void update(Vector2 mouse);
        void draw(Color color) const;
    };

    void regenerateAll();
    static void drawChannelRow(Rectangle bounds, const Channel& channel, double cursorTimeMs);
    void drawVerdictPanel(Rectangle bounds, const std::string& title, const std::string& formula,
                           int winner, const std::string& detail) const;

    Rectangle bounds_{};
    Rectangle channelBounds_[NUM_CHANNELS]{};
    Rectangle verdictBounds_[2]{};

    std::mt19937 rng_;
    std::vector<Channel> channels_;
    std::vector<Slider> sliders_;

    int firstSpikeWinner_ = -1;
    double firstSpikeWinnerTimeMs_ = -1.0;
    int rateArgmaxWinner_ = -1;
    double sweepProgress_ = 0.0;
    double cursorTimeMs_ = 0.0;
};
