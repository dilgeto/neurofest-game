#include "../include/network_view_module.hpp"
#include "../include/model_browser.hpp"
#include "../include/task_common.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    // Same playback speed levels as NeuroGame's "Evaluar red" (see main.cpp), cycled here by
    // clicking a per-panel button instead of the global SPACE key -- with up to 3 of these
    // running stacked at once, a single SPACE press would otherwise cycle all of them together.
    constexpr double SNN_SPEED_FACTORS[] = { 1.0, 20.0, 1000.0 };
    constexpr const char* SNN_SPEED_LABELS[] = { "Tiempo real", "Lenta", "Ultra lenta" };
    constexpr int SNN_SPEED_LEVEL_COUNT = 3;

    const char* taskName(int taskCategory) {
        switch (taskCategory) {
            case 0: return "Evaluar red: Acrobot";
            case 1: return "Evaluar red: Mountain Car";
            default: return "Evaluar red: Racing Car";
        }
    }

    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }
}

NetworkViewModule::NetworkViewModule(int taskCategory)
    : taskCategory_(taskCategory), name_(taskName(taskCategory)),
      speedButton_({0, 0, 1, 1}, ""), rng_(std::random_device{}()) {
    const SnnTaskCategory& category = snnTaskCategories()[static_cast<size_t>(taskCategory_)];
    std::vector<SnnModelEntry> models = listSnnModels(category.directory);
    if (!models.empty()) {
        outPath_ = models[0].outPath;
        wiPath_ = models[0].wiPath;
        loaded_ = true; // reloadNetwork() (called from setBounds()) does the actual load
    }
}

void NetworkViewModule::reloadNetwork() {
    if (!loaded_) return;
    const SnnTaskCategory& category = snnTaskCategories()[static_cast<size_t>(taskCategory_)];
    if (!network_.load(outPath_, wiPath_, category.preset(), networkPanelBounds_)) {
        loaded_ = false;
        return;
    }
    if (taskCategory_ == 0) {
        acrobotEnv_.reset(rng_);
        std::array<double, 6> obs = acrobotEnv_.observe();
        network_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
        pendingAction_ = network_.decodeFirstSpikeWinner() - 1.0;
    } else if (taskCategory_ == 1) {
        mountainCarEnv_.reset(rng_);
        std::array<double, 2> obs = mountainCarEnv_.observe();
        network_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
        pendingAction_ = network_.decodeFirstSpikeWinner() - 1.0;
    } else {
        carEnv_.reset(rng_);
        network_.simulateStep(encodeCarObservation(carEnv_.observe()));
        pendingThrottle_ = decodeCarContinuousAction(network_, 0);
        pendingSteering_ = decodeCarContinuousAction(network_, 1);
    }
}

void NetworkViewModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;
    // Network left half, task environment right half -- same split as NeuroGame's
    // "Evaluar red", just relative to this panel's bounds instead of the full window.
    float half = (bounds.width - 40.0f) / 2.0f;
    networkPanelBounds_ = { bounds.x, bounds.y + 40.0f, half, bounds.height - 40.0f };
    envPanelBounds_ = { bounds.x + half + 40.0f, bounds.y + 40.0f, half, bounds.height - 40.0f };
    speedButton_.setBounds({ bounds.x, bounds.y, 220.0f, 30.0f });
    speedButton_.setFontSize(FS(14));
    speedButton_.setText(std::string("Velocidad: ") + SNN_SPEED_LABELS[speedLevel_]);
    reloadNetwork(); // SnnNetwork::load() lays out its nodes against networkPanelBounds_
}

void NetworkViewModule::takeEnvStep() {
    if (taskCategory_ == 0) {
        acrobotEnv_.step(pendingAction_, rng_);
        std::array<double, 6> obs = acrobotEnv_.observe();
        network_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
        pendingAction_ = network_.decodeFirstSpikeWinner() - 1.0;
    } else if (taskCategory_ == 1) {
        mountainCarEnv_.step(pendingAction_, rng_);
        std::array<double, 2> obs = mountainCarEnv_.observe();
        network_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
        pendingAction_ = network_.decodeFirstSpikeWinner() - 1.0;
    } else {
        carEnv_.step(pendingThrottle_, pendingSteering_, rng_);
        network_.simulateStep(encodeCarObservation(carEnv_.observe()));
        pendingThrottle_ = decodeCarContinuousAction(network_, 0);
        pendingSteering_ = decodeCarContinuousAction(network_, 1);
    }
}

void NetworkViewModule::update(Vector2 mouse, float frameMs) {
    if (!loaded_) return;

    if (speedButton_.isClicked(mouse)) {
        speedLevel_ = (speedLevel_ + 1) % SNN_SPEED_LEVEL_COUNT;
        speedButton_.setText(std::string("Velocidad: ") + SNN_SPEED_LABELS[speedLevel_]);
    }

    double realStepMs = realStepMsForTask(taskCategory_) * SNN_SPEED_FACTORS[speedLevel_];
    double speed = SnnNetwork::SIM_WINDOW_MS / realStepMs; // sim-ms per wall-ms

    // A single advance() per rendered frame can't keep up with tasks whose real step rate
    // exceeds the render rate (Racing Car needs ~100 decisions/s) -- catch up with as many
    // steps as elapsed wall-clock time actually calls for, capped so a stall can't spiral.
    double remainingWallMs = std::min(static_cast<double>(frameMs), realStepMs * 10.0);
    while (remainingWallMs > 1e-9) {
        if (network_.isStepFinished()) takeEnvStep();
        double simMsLeftInWindow = (1.0 - network_.progress()) * SnnNetwork::SIM_WINDOW_MS;
        double wallMsThisIter = std::min(remainingWallMs, simMsLeftInWindow / speed);
        network_.advance(wallMsThisIter * speed);
        remainingWallMs -= wallMsThisIter;
    }
}

void NetworkViewModule::draw(Vector2 mouse) const {
    if (!loaded_) {
        const char* msg = "No hay modelos entrenados para esta tarea";
        int w = MeasureText(msg, FS(18));
        DrawText(msg, static_cast<int>(bounds_.x + bounds_.width / 2.0f - w / 2.0f),
            static_cast<int>(bounds_.y + bounds_.height / 2.0f), FS(18), GRAY);
        return;
    }

    // setIoDisplay's per-task input/output labels (mirrors main.cpp's updateSnnIoDisplay).
    if (taskCategory_ == 0) {
        std::array<double, 6> obs = acrobotEnv_.observe();
        std::vector<SnnIoEntry> inputs = {
            {"cos(theta1)", TextFormat("%.2f", obs[0])}, {"sin(theta1)", TextFormat("%.2f", obs[1])},
            {"cos(theta2)", TextFormat("%.2f", obs[2])}, {"sin(theta2)", TextFormat("%.2f", obs[3])},
            {"Vel. angular 1", TextFormat("%.2f rad/s", obs[4])}, {"Vel. angular 2", TextFormat("%.2f rad/s", obs[5])},
        };
        int winner = static_cast<int>(std::lround(pendingAction_ + 1.0));
        std::vector<SnnIoEntry> outputs = {
            {"Torque -1", (winner == 0) ? "ACTIVA" : ""}, {"Sin torque", (winner == 1) ? "ACTIVA" : ""},
            {"Torque +1", (winner == 2) ? "ACTIVA" : ""},
        };
        network_.setIoDisplay(inputs, outputs, winner);
    } else if (taskCategory_ == 1) {
        std::array<double, 2> obs = mountainCarEnv_.observe();
        std::vector<SnnIoEntry> inputs = { {"Posición", TextFormat("%.2f", obs[0])}, {"Velocidad", TextFormat("%.3f", obs[1])} };
        int winner = static_cast<int>(std::lround(pendingAction_ + 1.0));
        std::vector<SnnIoEntry> outputs = {
            {"Empuje izquierda", (winner == 0) ? "ACTIVA" : ""}, {"Sin empuje", (winner == 1) ? "ACTIVA" : ""},
            {"Empuje derecha", (winner == 2) ? "ACTIVA" : ""},
        };
        network_.setIoDisplay(inputs, outputs, winner);
    } else {
        std::array<double, 9> obs = carEnv_.observe();
        std::vector<SnnIoEntry> inputs = {
            {"Posición X", TextFormat("%.2f m", obs[0])}, {"Posición Y", TextFormat("%.2f m", obs[1])},
            {"Orientación", TextFormat("%.2f rad", obs[2])}, {"Vel. X", TextFormat("%.2f m/s", obs[3])},
            {"Vel. Y", TextFormat("%.2f m/s", obs[4])}, {"Vel. angular", TextFormat("%.2f rad/s", obs[5])},
            {"Lidar izquierdo", TextFormat("%.2f", obs[6])}, {"Lidar centro", TextFormat("%.2f", obs[7])},
            {"Lidar derecho", TextFormat("%.2f", obs[8])},
        };
        std::vector<SnnIoEntry> outputs = {
            {"Acelerador/Freno", TextFormat("%.2f", pendingThrottle_)}, {"Dirección", TextFormat("%.2f", pendingSteering_)},
        };
        network_.setIoDisplay(inputs, outputs, -1);
    }

    network_.draw(networkPanelBounds_);
    DrawText(name_.c_str(), static_cast<int>(networkPanelBounds_.x), static_cast<int>(bounds_.y + 10.0f), FS(18), DARKGRAY);

    if (taskCategory_ == 0) {
        acrobotEnv_.draw(envPanelBounds_, network_.progress());
        DrawText("Entorno: Acrobot", static_cast<int>(envPanelBounds_.x), static_cast<int>(bounds_.y + 10.0f), FS(18), DARKGRAY);
    } else if (taskCategory_ == 1) {
        mountainCarEnv_.draw(envPanelBounds_);
        DrawText("Entorno: Mountain Car", static_cast<int>(envPanelBounds_.x), static_cast<int>(bounds_.y + 10.0f), FS(18), DARKGRAY);
    } else {
        carEnv_.draw(envPanelBounds_);
        DrawText("Entorno: Racing Car", static_cast<int>(envPanelBounds_.x), static_cast<int>(bounds_.y + 10.0f), FS(18), DARKGRAY);
    }

    float dividerX = networkPanelBounds_.x + networkPanelBounds_.width + 20.0f;
    DrawLineEx({ dividerX, networkPanelBounds_.y }, { dividerX, networkPanelBounds_.y + networkPanelBounds_.height }, 1.0f, LIGHTGRAY);

    speedButton_.draw(mouse);
}
