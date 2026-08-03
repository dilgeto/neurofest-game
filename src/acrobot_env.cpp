#include "../include/acrobot_env.hpp"

#include <algorithm>
#include <cmath>

namespace {
    constexpr double LINK_LENGTH_1 = 1.0;
    constexpr double LINK_LENGTH_2 = 1.0;
    constexpr double LINK_MASS_1 = 1.0;
    constexpr double LINK_MASS_2 = 1.0;
    constexpr double LINK_COM_POS_1 = 0.5;
    constexpr double LINK_COM_POS_2 = 0.5;
    constexpr double LINK_MOI = 1.0;
    constexpr double MAX_VEL_1 = 4.0 * M_PI;
    constexpr double MAX_VEL_2 = 9.0 * M_PI;
    constexpr double MIN_TORQUE = -1.0;
    constexpr double MAX_TORQUE = 1.0;
    constexpr double GRAVITY = 9.8;

    double angleNormalize(double x) {
        double twoPi = 2.0 * M_PI;
        double a = x + M_PI;
        return (a - twoPi * std::floor(a / twoPi)) - M_PI;
    }

    void dsdt(const double state[4], double action, double dState[4]) {
        double theta1 = state[0];
        double theta2 = state[1];
        double dtheta1 = state[2];
        double dtheta2 = state[3];

        double d1 = LINK_MASS_1 * LINK_COM_POS_1 * LINK_COM_POS_1
            + LINK_MASS_2 * (LINK_LENGTH_1 * LINK_LENGTH_1 + LINK_COM_POS_2 * LINK_COM_POS_2
                + 2.0 * LINK_LENGTH_1 * LINK_COM_POS_2 * std::cos(theta2))
            + LINK_MOI + LINK_MOI;
        double d2 = LINK_MASS_2 * (LINK_COM_POS_2 * LINK_COM_POS_2 + LINK_LENGTH_1 * LINK_COM_POS_2 * std::cos(theta2)) + LINK_MOI;
        double phi2 = LINK_MASS_2 * LINK_COM_POS_2 * GRAVITY * std::cos(theta1 + theta2 - M_PI / 2.0);
        double phi1 = -LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2 * dtheta2 * dtheta2 * std::sin(theta2)
            - 2.0 * LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2 * dtheta2 * dtheta1 * std::sin(theta2)
            + (LINK_MASS_1 * LINK_COM_POS_1 + LINK_MASS_2 * LINK_LENGTH_1) * GRAVITY * std::cos(theta1 - M_PI / 2.0)
            + phi2;

        double ddtheta2 = (action + d2 / d1 * phi1 - LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2 * dtheta1 * dtheta1 * std::sin(theta2) - phi2)
            / (LINK_MASS_2 * LINK_COM_POS_2 * LINK_COM_POS_2 + LINK_MOI - d2 * d2 / d1);
        double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

        dState[0] = dtheta1;
        dState[1] = dtheta2;
        dState[2] = ddtheta1;
        dState[3] = ddtheta2;
    }

    void rk4(const double state[4], double action, double dt, double nextState[4]) {
        double k1[4], k2[4], k3[4], k4[4], y1[4], y2[4], y3[4];
        double dt2 = dt / 2.0;

        dsdt(state, action, k1);
        for (int i = 0; i < 4; ++i) y1[i] = state[i] + dt2 * k1[i];
        dsdt(y1, action, k2);
        for (int i = 0; i < 4; ++i) y2[i] = state[i] + dt2 * k2[i];
        dsdt(y2, action, k3);
        for (int i = 0; i < 4; ++i) y3[i] = state[i] + dt * k3[i];
        dsdt(y3, action, k4);
        for (int i = 0; i < 4; ++i) nextState[i] = state[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

void AcrobotEnv::reset(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    theta1_ = dist(rng);
    theta2_ = dist(rng);
    dtheta1_ = dist(rng);
    dtheta2_ = dist(rng);
    stepCount_ = 0;
}

std::array<double, 6> AcrobotEnv::observe() const {
    return {
        std::cos(theta1_), std::sin(theta1_),
        std::cos(theta2_), std::sin(theta2_),
        dtheta1_, dtheta2_
    };
}

void AcrobotEnv::step(double action, std::mt19937& rng) {
    double actionClamped = std::clamp(action, MIN_TORQUE, MAX_TORQUE);
    // rl-tools rescales actionClamped from [-1,1] to [MIN_TORQUE,MAX_TORQUE]; with the
    // symmetric ±1 range wann-cpp uses, that rescale is the identity.
    double actionScaled = (actionClamped + 1.0) / 2.0 * (MAX_TORQUE - MIN_TORQUE) + MIN_TORQUE;

    double state[4] = {theta1_, theta2_, dtheta1_, dtheta2_};
    double next[4];
    rk4(state, actionScaled, DT, next);

    theta1_ = angleNormalize(next[0]);
    theta2_ = angleNormalize(next[1]);
    dtheta1_ = std::clamp(next[2], -MAX_VEL_1, MAX_VEL_1);
    dtheta2_ = std::clamp(next[3], -MAX_VEL_2, MAX_VEL_2);
    ++stepCount_;

    bool terminated = (-std::cos(theta1_) - std::cos(theta2_ + theta1_)) > 1.0;
    if (terminated || stepCount_ >= EPISODE_STEP_LIMIT) {
        reset(rng);
    }
}

void AcrobotEnv::draw(Rectangle bounds) const {
    Vector2 pivot = { bounds.x + bounds.width / 2.0f, bounds.y + bounds.height * 0.32f };
    float scale = std::min(bounds.width, bounds.height) / 5.0f; // pixels per meter

    double mathX1 = LINK_LENGTH_1 * std::sin(theta1_);
    double mathY1 = -LINK_LENGTH_1 * std::cos(theta1_);
    Vector2 p1 = { pivot.x + static_cast<float>(mathX1) * scale, pivot.y - static_cast<float>(mathY1) * scale };

    double mathX2 = mathX1 + LINK_LENGTH_2 * std::sin(theta1_ + theta2_);
    double mathY2 = mathY1 - LINK_LENGTH_2 * std::cos(theta1_ + theta2_);
    Vector2 p2 = { pivot.x + static_cast<float>(mathX2) * scale, pivot.y - static_cast<float>(mathY2) * scale };

    DrawLineEx(pivot, p1, 6.0f, Color{60, 110, 220, 255});
    DrawLineEx(p1, p2, 6.0f, Color{220, 90, 70, 255});
    DrawCircleV(pivot, 7.0f, DARKGRAY);
    DrawCircleV(p1, 9.0f, Color{60, 110, 220, 255});
    DrawCircleV(p2, 9.0f, Color{220, 90, 70, 255});

    DrawText(TextFormat("Paso episodio: %d / %d", stepCount_, EPISODE_STEP_LIMIT),
        static_cast<int>(bounds.x), static_cast<int>(bounds.y + bounds.height - 30), 18, DARKGRAY);
}
