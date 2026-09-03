#include "../include/vs_ai_discrete_module.hpp"
#include "../include/model_browser.hpp"
#include "../include/task_common.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    const char* taskName(int taskCategory) { return taskCategory == 0 ? "VS IA: Acrobot" : "VS IA: Mountain Car"; }
    const char* modelDir(int taskCategory) { return taskCategory == 0 ? "models/acrobot" : "models/mountain_car"; }

    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }

    // Solid triangular direction arrow centered inside `bounds` (used on the square
    // izquierda/derecha buttons, which carry no text of their own) -- same helper as
    // main.cpp's drawArrowIcon.
    void drawArrowIcon(Rectangle bounds, bool pointsRight, Color color) {
        float cx = bounds.x + bounds.width / 2.0f;
        float cy = bounds.y + bounds.height / 2.0f;
        float half = std::min(bounds.width, bounds.height) * 0.28f;
        Vector2 tip = { cx + (pointsRight ? half : -half), cy };
        Vector2 baseNear = { cx + (pointsRight ? -half : half), cy - half };
        Vector2 baseFar = { cx + (pointsRight ? -half : half), cy + half };
        if (pointsRight) DrawTriangle(baseNear, baseFar, tip, color);
        else DrawTriangle(baseFar, baseNear, tip, color);
    }
}

VsAiDiscreteModule::VsAiDiscreteModule(int taskCategory)
    : taskCategory_(taskCategory), name_(taskName(taskCategory)),
      leftButton_({0, 0, 1, 1}, ""), rightButton_({0, 0, 1, 1}, ""), rng_(std::random_device{}()) {
    std::vector<SnnModelEntry> models = listSnnModels(modelDir(taskCategory_));
    if (models.empty()) return;
    const SnnTaskPreset& preset = (taskCategory_ == 0) ? snnAcrobotPreset() : snnMountainCarPreset();
    if (!aiNetwork_.load(models[0].outPath, models[0].wiPath, preset, {0, 0, 100, 100})) return;

    loaded_ = true;
    if (taskCategory_ == 0) {
        humanAcrobotEnv_.reset(rng_);
        aiAcrobotEnv_.reset(rng_);
        std::array<double, 6> obs = aiAcrobotEnv_.observe();
        aiNetwork_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
        aiAction_ = aiNetwork_.decodeFirstSpikeWinner() - 1.0;
    } else {
        humanMountainCarEnv_.reset(rng_);
        aiMountainCarEnv_.reset(rng_);
        std::array<double, 2> obs = aiMountainCarEnv_.observe();
        aiNetwork_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
        aiAction_ = aiNetwork_.decodeFirstSpikeWinner() - 1.0;
    }
}

void VsAiDiscreteModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;
    // Human's env left, AI's env right, leaving room at the bottom for the izquierda/derecha
    // buttons -- same split as main.cpp's discreteLeftPanelBounds/discreteRightPanelBounds.
    constexpr float CONTROLS_H = 130.0f;
    float envHeight = bounds.height - CONTROLS_H;
    float half = (bounds.width - 40.0f) / 2.0f;
    humanPanelBounds_ = { bounds.x, bounds.y + 30.0f, half, envHeight };
    aiPanelBounds_ = { bounds.x + half + 40.0f, bounds.y + 30.0f, half, envHeight };

    float controlsTop = bounds.y + bounds.height - CONTROLS_H + 20.0f;
    leftButton_.setBounds({ bounds.x + bounds.width / 2.0f - 110.0f, controlsTop, 100.0f, 100.0f });
    rightButton_.setBounds({ bounds.x + bounds.width / 2.0f + 10.0f, controlsTop, 100.0f, 100.0f });
}

void VsAiDiscreteModule::update(Vector2 mouse, float frameMs) {
    if (!loaded_) return;

    // Gamepad (analog): left stick X thresholded into a discrete left/right action.
    double humanAction = 0.0;
    if (IsGamepadAvailable(0)) {
        constexpr float STICK_DEADZONE = 0.12f;
        float stickX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X);
        if (stickX < -STICK_DEADZONE) humanAction = -1.0;
        else if (stickX > STICK_DEADZONE) humanAction = 1.0;
    }
    // Mouse (digital): overrides the gamepad whenever a button is actually held.
    if (leftButton_.isBeingClicked(mouse)) humanAction = -1.0;
    else if (rightButton_.isBeingClicked(mouse)) humanAction = 1.0;

    double physicsDtMs = realStepMsForTask(taskCategory_);
    accumulatorMs_ += frameMs;
    accumulatorMs_ = std::min(accumulatorMs_, physicsDtMs * 10.0);
    while (accumulatorMs_ >= physicsDtMs) {
        if (taskCategory_ == 0) {
            humanAcrobotEnv_.step(humanAction, rng_);
            aiAcrobotEnv_.step(aiAction_, rng_);
            std::array<double, 6> obs = aiAcrobotEnv_.observe();
            aiNetwork_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
            aiAction_ = aiNetwork_.decodeFirstSpikeWinner() - 1.0;
        } else {
            humanMountainCarEnv_.step(humanAction, rng_);
            aiMountainCarEnv_.step(aiAction_, rng_);
            std::array<double, 2> obs = aiMountainCarEnv_.observe();
            aiNetwork_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
            aiAction_ = aiNetwork_.decodeFirstSpikeWinner() - 1.0;
        }
        accumulatorMs_ -= physicsDtMs;
    }
}

void VsAiDiscreteModule::draw(Vector2 mouse) const {
    if (!loaded_) {
        const char* msg = "No hay modelos entrenados para esta tarea";
        int w = MeasureText(msg, FS(18));
        DrawText(msg, static_cast<int>(bounds_.x + bounds_.width / 2.0f - w / 2.0f),
            static_cast<int>(bounds_.y + bounds_.height / 2.0f), FS(18), GRAY);
        return;
    }

    double progress = accumulatorMs_ / realStepMsForTask(taskCategory_);
    if (taskCategory_ == 0) {
        humanAcrobotEnv_.draw(humanPanelBounds_, progress);
        aiAcrobotEnv_.draw(aiPanelBounds_, progress);
    } else {
        humanMountainCarEnv_.draw(humanPanelBounds_);
        aiMountainCarEnv_.draw(aiPanelBounds_);
    }

    DrawText(name_.c_str(), static_cast<int>(bounds_.x), static_cast<int>(bounds_.y + 5.0f), FS(18), DARKGRAY);
    DrawText("Tú", static_cast<int>(humanPanelBounds_.x), static_cast<int>(bounds_.y + 30.0f), FS(16), DARKGRAY);
    DrawText("IA", static_cast<int>(aiPanelBounds_.x), static_cast<int>(bounds_.y + 30.0f), FS(16), DARKGRAY);

    float dividerX = humanPanelBounds_.x + humanPanelBounds_.width + 20.0f;
    DrawLineEx({ dividerX, humanPanelBounds_.y }, { dividerX, humanPanelBounds_.y + humanPanelBounds_.height }, 1.0f, LIGHTGRAY);

    leftButton_.draw(mouse);
    rightButton_.draw(mouse);
    drawArrowIcon(leftButton_.getButton(), false, DARKGRAY);
    drawArrowIcon(rightButton_.getButton(), true, DARKGRAY);
}
