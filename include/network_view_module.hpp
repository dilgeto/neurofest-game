#pragma once
#include "acrobot_env.hpp"
#include "button.hpp"
#include "car_env.hpp"
#include "demo_module.hpp"
#include "mountain_car_env.hpp"
#include "snn_network.hpp"

#include <random>
#include <string>

// Extracted from main.cpp's NETWORK_VIEW ("Evaluar red"): SNN visualization + task
// environment side by side, in a real closed loop -- the network observes the environment,
// its decoded action drives the environment, repeat. Parameterized by task (0=Acrobot,
// 1=Mountain Car, 2=Racing Car) so all three can run as separate, simultaneously-selectable
// modules -- unlike the standalone NeuroGame flow, this auto-loads the first trained model
// found for its task (matching how VS AI mode already picks a model, since a per-panel
// "choose a model file" sub-menu would eat into an already-small stacked panel).
class NetworkViewModule : public IDemoModule {
public:
    explicit NetworkViewModule(int taskCategory); // 0=Acrobot, 1=Mountain Car, 2=Racing Car

    const char* name() const override { return name_.c_str(); }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    void reloadNetwork();     // (re)loads + relays out snnNetwork_ for the current bounds_
    void takeEnvStep();       // applies the pending action, observes, feeds the next window in

    int taskCategory_;
    std::string name_;
    std::string outPath_, wiPath_;
    bool loaded_ = false;

    Rectangle bounds_{};
    Rectangle networkPanelBounds_{};
    Rectangle envPanelBounds_{};
    Button speedButton_;

    // setIoDisplay() is called from draw() const to stage that frame's IO labels right before
    // rendering them -- transient display state, not simulated behavior, so mutable here is
    // the same judgment call as Button's draw()-time hover highlighting.
    mutable SnnNetwork network_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> unit_{0.0, 1.0};
    std::uniform_real_distribution<double> signedUnit_{-1.0, 1.0};

    AcrobotEnv acrobotEnv_;
    MountainCarEnv mountainCarEnv_;
    CarEnv carEnv_;
    double pendingAction_ = 0.0;
    double pendingThrottle_ = 0.0;
    double pendingSteering_ = 0.0;

    int speedLevel_ = 0; // index into SNN_SPEED_FACTORS/SNN_SPEED_LABELS (see .cpp)
};
