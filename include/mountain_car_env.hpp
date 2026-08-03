#pragma once
#include <raylib.h>
#include <array>
#include <random>

// Ports rl-tools' MountainCarContinuous environment (rl_tools/rl/environments/mountain_car)
// with the DiscMCParams override wann-cpp's SnnDiscMCTask.cpp uses (power=0.001,
// goal_position=0.5, EPISODE_STEPS=200). The "Mountain Car" models saved in this project
// are actually the discrete Mountain Car task (see model_browser.cpp's preset comments):
// same continuous physics, the action is just restricted to {-1, 0, +1}.
class MountainCarEnv {
    public:
        static constexpr double POWER = 0.001;
        static constexpr double GOAL_POSITION = 0.5;
        static constexpr double GOAL_VELOCITY = 0.0;
        static constexpr double MIN_POSITION = -1.2;
        static constexpr double MAX_POSITION = 0.6;
        static constexpr double MAX_SPEED = 0.07;
        static constexpr int EPISODE_STEP_LIMIT = 200;

        void reset(std::mt19937& rng);

        // Raw observation: [position, velocity]
        std::array<double, 2> observe() const;

        // Applies `force` (clamped to [-1,1]) for one physics tick. Auto-resets on
        // reaching the goal or hitting the step limit.
        void step(double force, std::mt19937& rng);

        void draw(Rectangle bounds) const;

    private:
        double position_ = -0.5;
        double velocity_ = 0.0;
        int stepCount_ = 0;
};
