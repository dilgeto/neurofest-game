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

        // Applies torque `action` (clamped to [-1,1]) for DT seconds, integrated as two
        // DT/2 RK4 sub-steps so the true (computed, not just interpolated) midpoint pose
        // is available for smoother animation via poseAt(). Auto-resets (new episode) if
        // the episode just terminated or hit the step limit.
        void step(double action, std::mt19937& rng);

        // Interpolates the pose through the start/midpoint/end keyframes of the most
        // recent step() (t=0 -> pre-step pose, t=0.5 -> the real RK4(DT/2) midpoint,
        // t=1 -> the post-step pose); t is clamped to [0,1].
        void poseAt(double t, double& theta1, double& theta2) const;

        void draw(Rectangle bounds, double t = 1.0) const;

    private:
        // theta1_/theta2_ are intentionally left unwrapped (not angle-normalized) between
        // steps: dsdt only ever consumes them through cos/sin, so wrapping would be purely
        // cosmetic while complicating the start/mid/end interpolation with a periodic
        // wrap-around case. observe() and terminated() are unaffected (also cos/sin-based).
        double theta1_ = 0.0, theta2_ = 0.0, dtheta1_ = 0.0, dtheta2_ = 0.0;
        double startTheta1_ = 0.0, startTheta2_ = 0.0;
        double midTheta1_ = 0.0, midTheta2_ = 0.0;
        int stepCount_ = 0;
};
