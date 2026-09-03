#pragma once
#include "acrobot_env.hpp"
#include "button.hpp"
#include "demo_module.hpp"
#include "mountain_car_env.hpp"
#include "snn_network.hpp"

#include <random>
#include <string>

// Extracted from main.cpp's VS_AI_ACROBOT / VS_AI_MOUNTAIN_CAR: human (on-screen
// izquierda/derecha buttons, or gamepad left stick) races a trained AI, side by side, on the
// discrete-action tasks. Parameterized by task (0=Acrobot, 1=Mountain Car) so both are
// separate, selectable modules; auto-loads the first trained model found (same as VS AI mode
// already does standalone).
class VsAiDiscreteModule : public IDemoModule {
public:
    explicit VsAiDiscreteModule(int taskCategory); // 0=Acrobot, 1=Mountain Car

    const char* name() const override { return name_.c_str(); }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    int taskCategory_;
    std::string name_;
    bool loaded_ = false;

    Rectangle bounds_{};
    Rectangle humanPanelBounds_{};
    Rectangle aiPanelBounds_{};
    Button leftButton_;
    Button rightButton_;

    std::mt19937 rng_;
    SnnNetwork aiNetwork_;
    AcrobotEnv humanAcrobotEnv_, aiAcrobotEnv_;
    MountainCarEnv humanMountainCarEnv_, aiMountainCarEnv_;
    double aiAction_ = 0.0;
    double accumulatorMs_ = 0.0;
};
