#pragma once
#include <raylib.h>
#include <array>
#include <random>

#include "iris_dataset.hpp"

// A single "round" of the Iris VS_AI head-to-head mode: a randomly drawn real flower
// from IRIS_DATASET, rendered as a stylized (non-diagnostic -- no color hints toward
// species) flower plus its 4 raw measurements. Unlike AcrobotEnv/MountainCarEnv/CarEnv
// there is no physics/step(): classification is a single shared decision, not a
// closed loop, so this only ever holds "the current flower".
class IrisEnv {
    public:
        // Draws a new random sample, distinct from the immediately previous one.
        void pickRandom(std::mt19937& rng);

        // TTFS-ready observation: the 4 raw features normalized to [0,1] via
        // IRIS_FEATURE_MIN/MAX (iris_dataset.hpp) -- same convention
        // encodeCarObservation (main.cpp) uses for Racing Car.
        std::array<double, 4> observe() const;

        int trueLabel() const { return sample_.label; }

        // Raw (un-normalized, cm) measurements of the current flower -- used to label the
        // 4 input neurons with their actual values in the "Evaluar red"-style IO display.
        const IrisSample& sample() const { return sample_; }

        void draw(Rectangle bounds) const;

    private:
        int sampleIndex_ = -1;
        IrisSample sample_{};
};
