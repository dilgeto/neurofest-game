#pragma once
#include "car_env.hpp"
#include "demo_module.hpp"
#include "snn_network.hpp"

#include <random>

// Extracted from main.cpp's VS_AI_RACING_CAR: human (arrow keys or gamepad: left stick
// steers, triggers accelerate/brake) races a trained AI on the same track. Auto-loads the
// first trained model found (same as VS AI mode already does standalone).
class VsAiRacingCarModule : public IDemoModule {
public:
    VsAiRacingCarModule();

    const char* name() const override { return "VS IA: Racing Car"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    bool loaded_ = false;
    Rectangle bounds_{};
    Rectangle trackBounds_{};

    std::mt19937 rng_;
    SnnNetwork aiNetwork_;
    CarEnv humanCarEnv_, aiCarEnv_;
    double aiThrottle_ = 0.0;
    double aiSteering_ = 0.0;
    double accumulatorMs_ = 0.0;
};
