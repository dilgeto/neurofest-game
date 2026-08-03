#pragma once
#include <raylib.h>
#include <array>
#include <random>

// Ports rl-tools' Acrobot environment (rl_tools/rl/environments/acrobot/operations_generic.h:
// dsdt/rk4/observe/terminated/sample_initial_state), with the MIN_TORQUE/MAX_TORQUE = -1/+1
// override wann-cpp's SnnAcrobotTask.cpp uses (GymAcrobotParams) -- with that symmetric
// range the rl-tools action rescale is the identity, so a decoded action in [-1,1] maps
// 1:1 to applied torque.
class AcrobotEnv {
    public:
        static constexpr double DT = 0.2;
        static constexpr int EPISODE_STEP_LIMIT = 500;

        void reset(std::mt19937& rng);

        // Raw observation: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2]
        std::array<double, 6> observe() const;

        // Applies torque `action` (clamped to [-1,1]) for DT seconds (RK4 integration).
        // Auto-resets (new episode) if the episode just terminated or hit the step limit.
        void step(double action, std::mt19937& rng);

        void draw(Rectangle bounds) const;

    private:
        double theta1_ = 0.0, theta2_ = 0.0, dtheta1_ = 0.0, dtheta2_ = 0.0;
        int stepCount_ = 0;
};
