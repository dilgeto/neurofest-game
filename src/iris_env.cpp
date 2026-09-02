#include "../include/iris_env.hpp"
#include "../include/ui_scale.hpp"

#include <algorithm>
#include <cmath>

namespace {
    constexpr int PETAL_COUNT = 6;
    constexpr int SEPAL_COUNT = 2;
    constexpr int CHAIN_SEGMENTS = 14;

    // Draws a petal/sepal as a chain of overlapping circles walking from `base` out
    // along `angle` for `length` pixels, radius tapering via sin(t*pi) (0 at the base
    // and tip, widest at the middle) so it reads as a rounded leaf shape.
    //
    // (DrawTriangleFan-based rotated-ellipse filling was tried first and rendered
    // nothing in this environment despite correct-looking geometry -- DrawCircleV is
    // used everywhere else in this project's *_env.cpp files and is known-reliable, so
    // this sticks to it instead of chasing the fan issue further.)
    void drawPetal(Vector2 base, float length, float maxWidth, float angle, Color color) {
        Vector2 dir = { std::cos(angle), std::sin(angle) };
        for (int i = 0; i <= CHAIN_SEGMENTS; ++i) {
            float t = static_cast<float>(i) / CHAIN_SEGMENTS;
            Vector2 p = { base.x + dir.x * length * t, base.y + dir.y * length * t };
            float radius = (maxWidth / 2.0f) * std::sin(t * PI);
            if (radius > 0.5f) DrawCircleV(p, radius, color);
        }
    }

    float normalize(float value, float minVal, float maxVal) {
        return std::clamp((value - minVal) / (maxVal - minVal), 0.0f, 1.0f);
    }
}

void IrisEnv::pickRandom(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, IRIS_SAMPLE_COUNT - 1);
    int next = sampleIndex_;
    while (next == sampleIndex_) next = dist(rng);
    sampleIndex_ = next;
    sample_ = IRIS_DATASET[static_cast<size_t>(sampleIndex_)];
}

std::array<double, 4> IrisEnv::observe() const {
    return {
        normalize(sample_.sepalLength, IRIS_FEATURE_MIN[0], IRIS_FEATURE_MAX[0]),
        normalize(sample_.sepalWidth, IRIS_FEATURE_MIN[1], IRIS_FEATURE_MAX[1]),
        normalize(sample_.petalLength, IRIS_FEATURE_MIN[2], IRIS_FEATURE_MAX[2]),
        normalize(sample_.petalWidth, IRIS_FEATURE_MIN[3], IRIS_FEATURE_MAX[3]),
    };
}

void IrisEnv::draw(Rectangle bounds) const {
    const int FONT = static_cast<int>(std::lround(18.0f * g_uiScale));
    DrawText(TextFormat("Sepalo: %.1f x %.1f cm", sample_.sepalLength, sample_.sepalWidth),
        static_cast<int>(bounds.x), static_cast<int>(bounds.y), FONT, DARKGRAY);
    DrawText(TextFormat("Petalo: %.1f x %.1f cm", sample_.petalLength, sample_.petalWidth),
        static_cast<int>(bounds.x), static_cast<int>(bounds.y + FONT + 6), FONT, DARKGRAY);

    Vector2 center = { bounds.x + bounds.width / 2.0f, bounds.y + bounds.height * 0.58f };
    // Each petal/sepal chain walks from the center out to 2*radiusA (drawPetal's
    // `length` argument), so maxReach is kept well under half the panel's short side.
    float maxReach = std::min(bounds.width, bounds.height) * 0.22f;

    // Sepals: shorter reach than the petals so they read as peeking out from behind.
    float sepalLenT = normalize(sample_.sepalLength, IRIS_FEATURE_MIN[0], IRIS_FEATURE_MAX[0]);
    float sepalWidT = normalize(sample_.sepalWidth, IRIS_FEATURE_MIN[1], IRIS_FEATURE_MAX[1]);
    float sepalA = maxReach * (0.30f + 0.30f * sepalLenT);
    float sepalB = maxReach * (0.06f + 0.08f * sepalWidT);
    Color sepalColor = Color{90, 150, 90, 255};
    for (int i = 0; i < SEPAL_COUNT; ++i) {
        float angle = (2.0f * PI / PETAL_COUNT) * (0.5f + static_cast<float>(i) * (PETAL_COUNT / static_cast<float>(SEPAL_COUNT)));
        drawPetal(center, sepalA * 2.0f, sepalB * 2.0f, angle, sepalColor);
    }

    // Petals: radiate evenly around the center; size encodes petal length/width.
    float petalLenT = normalize(sample_.petalLength, IRIS_FEATURE_MIN[2], IRIS_FEATURE_MAX[2]);
    float petalWidT = normalize(sample_.petalWidth, IRIS_FEATURE_MIN[3], IRIS_FEATURE_MAX[3]);
    float petalA = maxReach * (0.35f + 0.65f * petalLenT);
    float petalB = maxReach * (0.12f + 0.20f * petalWidT);
    Color petalColor = Color{190, 140, 210, 255};
    for (int i = 0; i < PETAL_COUNT; ++i) {
        float angle = (2.0f * PI / PETAL_COUNT) * static_cast<float>(i);
        drawPetal(center, petalA * 2.0f, petalB * 2.0f, angle, petalColor);
    }

    DrawCircleV(center, maxReach * 0.14f, Color{230, 200, 80, 255});
    DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), maxReach * 0.14f, Fade(BLACK, 0.3f));
}
