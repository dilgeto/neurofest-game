#include "../include/mountain_car_env.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    double heightAt(double x) {
        return std::sin(3.0 * x) * 0.45 + 0.55;
    }
}

void MountainCarEnv::reset(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-0.6, -0.4);
    position_ = dist(rng);
    velocity_ = 0.0;
    stepCount_ = 0;
}

std::array<double, 2> MountainCarEnv::observe() const {
    return { position_, velocity_ };
}

void MountainCarEnv::step(double force, std::mt19937& rng) {
    double forceClamped = std::clamp(force, -1.0, 1.0);
    velocity_ += forceClamped * POWER - 0.0025 * std::cos(3.0 * position_);
    velocity_ = std::clamp(velocity_, -MAX_SPEED, MAX_SPEED);
    position_ += velocity_;
    position_ = std::clamp(position_, MIN_POSITION, MAX_POSITION);
    if (position_ == MIN_POSITION && velocity_ < 0.0) velocity_ = 0.0;
    ++stepCount_;

    bool terminated = position_ >= GOAL_POSITION && velocity_ >= GOAL_VELOCITY;
    if (terminated || stepCount_ >= EPISODE_STEP_LIMIT) reset(rng);
}

void MountainCarEnv::draw(Rectangle bounds) const {
    constexpr int SEGMENTS = 100;
    Vector2 prev{};
    for (int i = 0; i <= SEGMENTS; ++i) {
        double x = MIN_POSITION + (MAX_POSITION - MIN_POSITION) * i / SEGMENTS;
        double h = heightAt(x);
        float sx = bounds.x + static_cast<float>((x - MIN_POSITION) / (MAX_POSITION - MIN_POSITION)) * bounds.width;
        float sy = bounds.y + bounds.height - static_cast<float>(h) * bounds.height * 0.85f;
        Vector2 p{ sx, sy };
        if (i > 0) DrawLineEx(prev, p, 3.0f, Color{90, 140, 90, 255});
        prev = p;
    }

    // Goal flag
    double goalHeight = heightAt(GOAL_POSITION);
    float goalX = bounds.x + static_cast<float>((GOAL_POSITION - MIN_POSITION) / (MAX_POSITION - MIN_POSITION)) * bounds.width;
    float goalY = bounds.y + bounds.height - static_cast<float>(goalHeight) * bounds.height * 0.85f;
    DrawLineEx({ goalX, goalY }, { goalX, goalY - 30 }, 2.0f, DARKGRAY);
    DrawTriangle({ goalX, goalY - 30 }, { goalX, goalY - 18 }, { goalX + 16, goalY - 24 }, Color{220, 90, 70, 255});

    // Car
    double carHeight = heightAt(position_);
    float carX = bounds.x + static_cast<float>((position_ - MIN_POSITION) / (MAX_POSITION - MIN_POSITION)) * bounds.width;
    float carY = bounds.y + bounds.height - static_cast<float>(carHeight) * bounds.height * 0.85f;
    DrawCircleV({ carX, carY - 8 }, 10.0f, Color{60, 110, 220, 255});

    DrawText(TextFormat("Paso episodio: %d / %d", stepCount_, EPISODE_STEP_LIMIT),
        static_cast<int>(bounds.x), static_cast<int>(bounds.y + bounds.height - 20),
        static_cast<int>(std::lround(18.0f * g_uiScale)), DARKGRAY);
}
