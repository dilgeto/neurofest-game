#include "../include/vs_ai_iris_module.hpp"
#include "../include/model_browser.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    constexpr double ANIM_DURATION_MS = 900.0; // wall-clock time to play one 20ms window
    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }
}

VsAiIrisModule::VsAiIrisModule()
    : guessSetosa_({0, 0, 1, 1}, "Setosa"), guessVersicolor_({0, 0, 1, 1}, "Versicolor"),
      guessVirginica_({0, 0, 1, 1}, "Virginica"), nextRound_({0, 0, 1, 1}, "Siguiente flor"),
      rng_(std::random_device{}()) {
    std::vector<SnnModelEntry> models = listSnnModels("models/iris");
    if (models.empty()) return;
    outPath_ = models[0].outPath;
    wiPath_ = models[0].wiPath;
    if (!network_.load(outPath_, wiPath_, snnIrisPreset(), {0, 0, 100, 100})) return;
    loaded_ = true;
    startRound();
}

void VsAiIrisModule::startRound() {
    round_.pickRandom(rng_);
    std::array<double, 4> obs = round_.observe();
    network_.simulateStep(std::vector<double>(obs.begin(), obs.end()));
    humanGuess_ = -1;
    aiGuess_ = -1;
    roundStartTime_ = GetTime();
    phase_ = Phase::Deciding;
}

void VsAiIrisModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;

    constexpr float BUTTONS_H = 70.0f;
    constexpr float TEXT_H = 30.0f;
    float envHeight = bounds.height - BUTTONS_H - TEXT_H - 40.0f;
    float half = (bounds.width - 40.0f) / 2.0f;
    networkPanelBounds_ = { bounds.x, bounds.y + 40.0f, half, envHeight };
    flowerPanelBounds_ = { bounds.x + half + 40.0f, bounds.y + 40.0f, half, envHeight };

    float buttonsY = networkPanelBounds_.y + networkPanelBounds_.height + 10.0f;
    guessSetosa_.setBounds({ bounds.x + bounds.width / 2.0f - 320.0f, buttonsY, 200.0f, 60.0f });
    guessVersicolor_.setBounds({ bounds.x + bounds.width / 2.0f - 100.0f, buttonsY, 200.0f, 60.0f });
    guessVirginica_.setBounds({ bounds.x + bounds.width / 2.0f + 120.0f, buttonsY, 200.0f, 60.0f });
    nextRound_.setBounds({ bounds.x + bounds.width / 2.0f - 110.0f, buttonsY, 220.0f, 60.0f });

    if (loaded_) {
        network_.load(outPath_, wiPath_, snnIrisPreset(), networkPanelBounds_); // relays out nodes for the new bounds
        startRound();
    }
}

void VsAiIrisModule::update(Vector2 mouse, float frameMs) {
    if (!loaded_) return;

    if (phase_ == Phase::Deciding) {
        if (!network_.isStepFinished()) {
            double speed = SnnNetwork::SIM_WINDOW_MS / ANIM_DURATION_MS;
            network_.advance(frameMs * speed);
        }

        if (humanGuess_ < 0) {
            if (guessSetosa_.isClicked(mouse)) humanGuess_ = 0;
            else if (guessVersicolor_.isClicked(mouse)) humanGuess_ = 1;
            else if (guessVirginica_.isClicked(mouse)) humanGuess_ = 2;
            if (humanGuess_ >= 0) humanDecisionTimeSec_ = GetTime() - roundStartTime_;
        }

        if (humanGuess_ >= 0 && network_.isStepFinished()) {
            aiGuess_ = network_.decodeFirstSpikeWinner();
            bool humanCorrect = humanGuess_ == round_.trueLabel();
            bool aiCorrect = aiGuess_ == round_.trueLabel();
            if (humanCorrect) { ++humanScore_; ++streak_; } else { streak_ = 0; }
            if (aiCorrect) ++aiScore_;
            phase_ = Phase::Revealing;
        }
    } else if (nextRound_.isClicked(mouse)) {
        startRound();
    }
}

void VsAiIrisModule::draw(Vector2 mouse) const {
    if (!loaded_) {
        const char* msg = "No hay modelos entrenados para esta tarea";
        int w = MeasureText(msg, FS(18));
        DrawText(msg, static_cast<int>(bounds_.x + bounds_.width / 2.0f - w / 2.0f),
            static_cast<int>(bounds_.y + bounds_.height / 2.0f), FS(18), GRAY);
        return;
    }

    int winnerHighlight = (phase_ == Phase::Revealing) ? aiGuess_ : -1;
    const IrisSample& sample = round_.sample();
    std::vector<SnnIoEntry> inputs = {
        {"Sépalo largo", TextFormat("%.1f cm", sample.sepalLength)}, {"Sépalo ancho", TextFormat("%.1f cm", sample.sepalWidth)},
        {"Pétalo largo", TextFormat("%.1f cm", sample.petalLength)}, {"Pétalo ancho", TextFormat("%.1f cm", sample.petalWidth)},
    };
    std::vector<SnnIoEntry> outputs = {
        {"Setosa", (winnerHighlight == 0) ? "ACTIVA" : ""}, {"Versicolor", (winnerHighlight == 1) ? "ACTIVA" : ""},
        {"Virginica", (winnerHighlight == 2) ? "ACTIVA" : ""},
    };
    network_.setIoDisplay(inputs, outputs, winnerHighlight);
    network_.draw(networkPanelBounds_);
    DrawText("VS IA: Iris (red)", static_cast<int>(bounds_.x), static_cast<int>(bounds_.y + 10.0f), FS(16), DARKGRAY);

    round_.draw(flowerPanelBounds_);
    DrawText("Adivina la flor", static_cast<int>(flowerPanelBounds_.x), static_cast<int>(bounds_.y + 10.0f), FS(16), DARKGRAY);

    float dividerX = networkPanelBounds_.x + networkPanelBounds_.width + 20.0f;
    DrawLineEx({ dividerX, networkPanelBounds_.y }, { dividerX, networkPanelBounds_.y + networkPanelBounds_.height }, 1.0f, LIGHTGRAY);

    std::string scoreText = TextFormat("Tú: %d   IA: %d   Racha: %d", humanScore_, aiScore_, streak_);
    int scoreWidth = MeasureText(scoreText.c_str(), FS(16));
    DrawText(scoreText.c_str(), static_cast<int>(bounds_.x + bounds_.width / 2.0f - scoreWidth / 2.0f),
        static_cast<int>(bounds_.y + 10.0f), FS(16), DARKGRAY);

    if (phase_ == Phase::Deciding) {
        guessSetosa_.draw(mouse);
        guessVersicolor_.draw(mouse);
        guessVirginica_.draw(mouse);

        double elapsedSec = (humanGuess_ < 0) ? (GetTime() - roundStartTime_) : humanDecisionTimeSec_;
        std::string prompt = (humanGuess_ < 0)
            ? TextFormat("Elige una especie mientras la red decide... (%.1fs)", elapsedSec)
            : TextFormat("Elegiste en %.1fs. Esperando a la IA...", elapsedSec);
        int promptWidth = MeasureText(prompt.c_str(), FS(14));
        DrawText(prompt.c_str(), static_cast<int>(bounds_.x + bounds_.width / 2.0f - promptWidth / 2.0f),
            static_cast<int>(guessSetosa_.getButton().y + guessSetosa_.getButton().height + 8.0f), FS(14), GRAY);
    } else {
        bool humanCorrect = humanGuess_ == round_.trueLabel();
        bool aiCorrect = aiGuess_ == round_.trueLabel();
        std::string reveal = TextFormat("Real: %s | Vos: %s %s en %.1fs | IA: %s %s",
            IRIS_SPECIES_NAMES[round_.trueLabel()],
            IRIS_SPECIES_NAMES[humanGuess_], humanCorrect ? "(correcto)" : "(incorrecto)", humanDecisionTimeSec_,
            IRIS_SPECIES_NAMES[aiGuess_], aiCorrect ? "(correcto)" : "(incorrecto)");
        int revealWidth = MeasureText(reveal.c_str(), FS(15));
        DrawText(reveal.c_str(), static_cast<int>(bounds_.x + bounds_.width / 2.0f - revealWidth / 2.0f),
            static_cast<int>(nextRound_.getButton().y - 26.0f), FS(15),
            humanCorrect ? Color{40, 150, 70, 255} : Color{200, 60, 60, 255});
        nextRound_.draw(mouse);
    }
}
