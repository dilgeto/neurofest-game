#pragma once
#include "button.hpp"
#include "demo_module.hpp"
#include "wann/Ind.h"
#include "wann/Hyperparams.h"
#include "wann/Wann.h"

#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>

// Extracted from main_wann_evolution_demo.cpp: runs a *live* WANN (Weight Agnostic Neural
// Network) evolution -- the real algorithm vendored from ../wann-cpp, unmodified -- against a
// small 2D "point inside circle" classification task, and animates it automatically, forever,
// with no input required, in a 2x2 panel grid (champion network / decision boundary /
// population grid / fitness history). See the original file's header comment for the full
// explanation; unchanged here. Unlike the other modules, this one fully self-paces and
// self-restarts (see AUTO_RESET_* in the .cpp) so it can be left unattended -- a natural fit
// for one slot in MultiDemo's combined view.
class WannEvolutionModule : public IDemoModule {
public:
    WannEvolutionModule();
    ~WannEvolutionModule() override;
    WannEvolutionModule(const WannEvolutionModule&) = delete;
    WannEvolutionModule& operator=(const WannEvolutionModule&) = delete;

    const char* name() const override { return "Neuroevolucion WANN"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

    // Public so the free helper functions in wann_evolution_module.cpp's anonymous namespace
    // (e.g. evalPopulation) can take it by const reference.
    struct ClassificationTask {
        std::vector<double> xs, ys;
        std::vector<int> labels;
        void resample(std::mt19937& rng);
    };

private:
    struct VisNode { int id; int type; float depth; };
    struct VisEdge { int srcId; int dstId; bool enabled; bool excitatory; };

    struct ChampionLayout {
        std::vector<VisNode> nodes;
        std::vector<VisEdge> edges;
        double bestWeightVal = 1.0;
        int nEnabledConns = 0;
    };

    struct Slider {
        Rectangle track{};
        float minValue = 0.0f, maxValue = 1.0f, value = 0.0f;
        bool dragging = false;
        void update(Vector2 mouse);
        void draw() const;
    };

    void resetEvolution();
    void advanceGeneration();
    ChampionLayout computeChampionLayout(const wann::Ind& ind, const std::vector<double>& reward) const;

    void drawNetworkPanel() const;
    void drawBoundaryPanel() const;
    void drawPopulationPanel() const;
    void drawChartPanel() const;

    Rectangle bounds_{};
    Rectangle networkPanel_{}, boundaryPanel_{}, popPanel_{}, chartPanel_{};
    float topBarY_ = 0.0f;
    Button playPauseButton_;
    Button resetButton_;
    Slider speedSlider_;

    std::mt19937 seedRng_;
    wann::Hyperparams hyp_;
    ClassificationTask task_;
    wann::Wann* wannAlgo_ = nullptr;

    int generation_ = 0;
    double bestFitness_ = 0.0, meanFitness_ = 0.0;
    std::vector<double> popFitness_;
    ChampionLayout championLayout_;
    std::vector<float> boundaryPixels_;
    Texture2D boundaryTex_{};
    std::deque<float> bestHistory_, meanHistory_;

    std::unordered_map<int, double> nodeFirstSeen_;
    std::unordered_map<int64_t, double> edgeFirstSeen_;
    double solvedHoldTimer_ = 0.0;

    bool running_ = true;
    double accumulatorSec_ = 0.0;
};
