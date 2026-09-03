#include "../include/vs_ai_racing_car_module.hpp"
#include "../include/model_browser.hpp"
#include "../include/task_common.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }
    const Color HUMAN_CAR_COLOR = Color{220, 60, 60, 255};
    const Color AI_CAR_COLOR = Color{60, 110, 220, 255};
}

VsAiRacingCarModule::VsAiRacingCarModule() : rng_(std::random_device{}()) {
    std::vector<SnnModelEntry> models = listSnnModels("models/racing_car");
    if (models.empty()) return;
    if (!aiNetwork_.load(models[0].outPath, models[0].wiPath, snnRacingCarPreset(), {0, 0, 100, 100})) return;

    loaded_ = true;
    humanCarEnv_.reset(rng_);
    aiCarEnv_.reset(rng_);
    aiNetwork_.simulateStep(encodeCarObservation(aiCarEnv_.observe()));
    aiThrottle_ = decodeCarContinuousAction(aiNetwork_, 0);
    aiSteering_ = decodeCarContinuousAction(aiNetwork_, 1);
}

void VsAiRacingCarModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;
    trackBounds_ = { bounds.x, bounds.y + 40.0f, bounds.width, bounds.height - 70.0f };
}

void VsAiRacingCarModule::update(Vector2 /*mouse*/, float frameMs) {
    if (!loaded_) return;

    double humanThrottle = 0.0;
    double humanSteering = 0.0;

    // Gamepad (analog): left stick steers, triggers accelerate/brake.
    if (IsGamepadAvailable(0)) {
        constexpr float STICK_DEADZONE = 0.12f;
        float stickX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X);
        if (stickX < -STICK_DEADZONE || stickX > STICK_DEADZONE) {
            humanSteering = -static_cast<double>(stickX);
        }
        float rightTrigger = GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_TRIGGER);
        float leftTrigger = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_TRIGGER);
        double accel = std::clamp((static_cast<double>(rightTrigger) + 1.0) / 2.0, 0.0, 1.0);
        double brake = std::clamp((static_cast<double>(leftTrigger) + 1.0) / 2.0, 0.0, 1.0);
        humanThrottle = accel - brake;
    }

    // Keyboard (digital): overrides the gamepad whenever a relevant key is actually held.
    if (IsKeyDown(KEY_UP)) humanThrottle = 1.0;
    else if (IsKeyDown(KEY_DOWN)) humanThrottle = -1.0;
    if (IsKeyDown(KEY_LEFT)) humanSteering = 1.0;
    else if (IsKeyDown(KEY_RIGHT)) humanSteering = -1.0;

    humanThrottle = std::clamp(humanThrottle, -1.0, 1.0);
    humanSteering = std::clamp(humanSteering, -1.0, 1.0);

    const double physicsDtMs = CarEnv::DT * 1000.0;
    accumulatorMs_ += frameMs;
    accumulatorMs_ = std::min(accumulatorMs_, physicsDtMs * 10.0);
    while (accumulatorMs_ >= physicsDtMs) {
        humanCarEnv_.step(humanThrottle, humanSteering, rng_);

        aiCarEnv_.step(aiThrottle_, aiSteering_, rng_);
        aiNetwork_.simulateStep(encodeCarObservation(aiCarEnv_.observe()));
        aiThrottle_ = decodeCarContinuousAction(aiNetwork_, 0);
        aiSteering_ = decodeCarContinuousAction(aiNetwork_, 1);

        accumulatorMs_ -= physicsDtMs;
    }
}

void VsAiRacingCarModule::draw(Vector2 /*mouse*/) const {
    if (!loaded_) {
        const char* msg = "No hay modelos entrenados para esta tarea";
        int w = MeasureText(msg, FS(18));
        DrawText(msg, static_cast<int>(bounds_.x + bounds_.width / 2.0f - w / 2.0f),
            static_cast<int>(bounds_.y + bounds_.height / 2.0f), FS(18), GRAY);
        return;
    }

    aiCarEnv_.drawTrack(trackBounds_);
    humanCarEnv_.drawCar(trackBounds_, HUMAN_CAR_COLOR, false);
    aiCarEnv_.drawCar(trackBounds_, AI_CAR_COLOR, true);

    DrawText("Tú (rojo) vs IA (azul)", static_cast<int>(bounds_.x), static_cast<int>(bounds_.y + 8.0f), FS(18), DARKGRAY);
    DrawText("Flechas o gamepad: arriba/abajo o gatillos acelerar/frenar, izquierda/derecha o stick girar",
        static_cast<int>(bounds_.x), static_cast<int>(trackBounds_.y + trackBounds_.height + 6.0f), FS(14), GRAY);
}
