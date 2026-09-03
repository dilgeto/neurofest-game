#pragma once
#include "button.hpp"
#include "demo_module.hpp"
#include "iris_env.hpp"
#include "snn_network.hpp"

#include <random>
#include <string>

// Extracted from main.cpp's VS_AI_IRIS: a single shared flower -- the human clicks a species
// while the AI's network plays out its 20ms decision window, then both guesses are revealed
// together against the flower's real species. Auto-loads the first trained model found (same
// as VS AI mode already does standalone).
class VsAiIrisModule : public IDemoModule {
public:
    VsAiIrisModule();

    const char* name() const override { return "VS IA: Iris"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    enum class Phase { Deciding, Revealing };

    void startRound();

    bool loaded_ = false;
    std::string outPath_, wiPath_;
    Rectangle bounds_{};
    Rectangle networkPanelBounds_{};
    Rectangle flowerPanelBounds_{};

    Button guessSetosa_, guessVersicolor_, guessVirginica_, nextRound_;

    std::mt19937 rng_;
    // setIoDisplay() is called from draw() const to stage that frame's IO labels right before
    // rendering them -- transient display state, not simulated behavior, so mutable here is
    // the same judgment call as Button's draw()-time hover highlighting.
    mutable SnnNetwork network_;
    IrisEnv round_;
    Phase phase_ = Phase::Deciding;
    int humanGuess_ = -1;
    int aiGuess_ = -1;
    int humanScore_ = 0;
    int aiScore_ = 0;
    int streak_ = 0;
    double roundStartTime_ = 0.0;
    double humanDecisionTimeSec_ = 0.0;
};
