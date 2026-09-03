#include <raylib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/branding.hpp"
#include "../include/button.hpp"
#include "../include/ui_scale.hpp"
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Random.h"
#include "../include/wann/Wann.h"

// Standalone demo (separate binary from NeuroGame): runs a *live* WANN (Weight Agnostic
// Neural Network) evolution -- using the real algorithm vendored from ../wann-cpp
// (Ind.cpp/Wann.cpp/NsgaSort.cpp, unmodified) -- against a small 2D "point inside circle"
// classification task, and animates it automatically, forever, with no input required. Runs
// at 1920x1080 (unlike the other demos here, which target the festival's dual-4K portrait
// kiosk) in a 2x2 panel grid, sped up well past real-time so a full run is a ~10-20 second
// demo reel rather than something a viewer has to stand and wait for:
//
//   1. Champion network (top-left): the best individual's topology, redrawn every
//      generation. New hidden nodes/connections pop in with a flash so a viewer can see
//      *where* the last mutation landed; connections WANN disabled are drawn as faint grey
//      dashes rather than removed (the algorithm keeps disabled genes, never deletes them).
//   2. Decision boundary (top-right): what the champion's classifier actually looks like in
//      input space -- blue/orange background = its prediction, dots = the true labels, so a
//      viewer can see accuracy directly instead of reading a number.
//   3. Population grid (bottom-left): every individual this generation as one tile, colour =
//      that individual's own fitness (dark purple -> teal -> gold), so a viewer can see the
//      whole population's fitness distribution shift generation to generation.
//   4. Fitness history (bottom-right): best/mean fitness across generations so far.
//
// Play/Pause + a speed slider are provided (matching the other demos' control bar) but the
// run also fully self-paces and self-restarts (see AUTO_RESET_* below) so it can be left
// unattended.

namespace {
    constexpr int SCREEN_WIDTH  = 1920;
    constexpr int SCREEN_HEIGHT = 1080;

    // --- WANN population / task ---
    constexpr int    POP_SIZE          = 60;
    constexpr int    N_TASK_POINTS     = 140;
    constexpr double  CIRCLE_R2        = 0.64; // radius 0.8 in a [-1,1]^2 box (~50/50 split)
    const std::vector<double> WEIGHT_VALS = { -2.0, -1.0, -0.5, 0.5, 1.0, 2.0 };

    // --- pacing / auto-restart ---
    // This is a demo reel, not a real-time simulation: default pace blows through a full run
    // (~25-180 generations) in well under half a minute. The slider still allows slowing down
    // to something inspectable if someone wants to linger on a particular generation.
    constexpr double DEFAULT_GENS_PER_SEC = 12.0;
    constexpr double MIN_GENS_PER_SEC     = 1.0;
    constexpr double MAX_GENS_PER_SEC     = 50.0;
    constexpr int    AUTO_RESET_MAX_GEN   = 180;
    constexpr int    AUTO_RESET_MIN_GEN   = 25;
    constexpr double AUTO_RESET_FITNESS   = 0.97;
    constexpr double SOLVED_HOLD_SECONDS  = 2.0;

    // --- champion diagram animation ---
    // Short on purpose: at the default pace generations land every ~0.08s, so a long pop-in
    // would just permanently overlap itself instead of reading as discrete "something changed"
    // flashes.
    constexpr double POP_IN_SECONDS = 0.22;

    // Shared palette (matches snn_network.cpp so the whole festival suite reads the same way).
    const Color COLOR_BIAS       = GRAY;
    const Color COLOR_INPUT      = Color{80, 140, 230, 255};
    const Color COLOR_OUTPUT     = Color{240, 170, 60, 255};
    const Color COLOR_HIDDEN     = Color{170, 120, 220, 255};
    const Color COLOR_EXCITATORY = Color{60, 200, 120, 255};
    const Color COLOR_INHIBITORY = Color{220, 70, 70, 255};
    const Color COLOR_CLASS0     = Color{50, 120, 225, 255};  // blue
    const Color COLOR_CLASS1     = Color{240, 140, 40, 255};  // orange
    const Color FIT_LOW          = Color{48, 30, 80, 255};    // dark purple
    const Color FIT_MID          = Color{34, 140, 150, 255};  // teal
    const Color FIT_HIGH         = Color{255, 205, 60, 255};  // gold

    int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }

    Color lerpColor(Color a, Color b, float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        return Color{
            static_cast<unsigned char>(a.r + (b.r - a.r) * t),
            static_cast<unsigned char>(a.g + (b.g - a.g) * t),
            static_cast<unsigned char>(a.b + (b.b - a.b) * t),
            static_cast<unsigned char>(a.a + (b.a - a.a) * t)
        };
    }

    Color fitnessColor(double fitness) {
        float t = static_cast<float>(std::clamp(fitness, 0.0, 1.0));
        return (t < 0.5f) ? lerpColor(FIT_LOW, FIT_MID, t / 0.5f)
                           : lerpColor(FIT_MID, FIT_HIGH, (t - 0.5f) / 0.5f);
    }

    float smoothstep01(double t) {
        float x = static_cast<float>(std::clamp(t, 0.0, 1.0));
        return x * x * (3.0f - 2.0f * x);
    }

    // -------------------------------------------------------------------
    // Task: classify whether (x, y) in [-1,1]^2 lies inside a circle.
    // Chosen because WANN's Gaussian/Squared activations make it directly
    // learnable by topology + activation choice alone, which is the point
    // WANN is demonstrating (shared-weight networks, no weight training).
    // -------------------------------------------------------------------
    struct ClassificationTask {
        std::vector<double> xs, ys;
        std::vector<int> labels;

        void resample(std::mt19937& rng) {
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            xs.assign(N_TASK_POINTS, 0.0);
            ys.assign(N_TASK_POINTS, 0.0);
            labels.assign(N_TASK_POINTS, 0);
            for (int i = 0; i < N_TASK_POINTS; ++i) {
                double x = dist(rng), y = dist(rng);
                xs[i] = x; ys[i] = y;
                labels[i] = (x * x + y * y < CIRCLE_R2) ? 1 : 0;
            }
        }
    };

    // Runs the champion's network on a single (x, y) at a given shared weight value.
    double evalNetOutput(const wann::Ind& ind, double weightVal, double x, double y) {
        std::vector<double> wMat = wann::setWeights(ind.wVec, weightVal);
        std::vector<double> out = wann::act(wMat, ind.aVec, ind.nInput, ind.nOutput, {x, y});
        return out[0];
    }

    // reward[i][j] = accuracy of individual i at shared weight WEIGHT_VALS[j].
    std::vector<std::vector<double>> evalPopulation(std::vector<wann::Ind>& pop, const ClassificationTask& task) {
        std::vector<std::vector<double>> reward(pop.size());
        for (size_t i = 0; i < pop.size(); ++i) {
            wann::Ind& ind = pop[i];
            reward[i].assign(WEIGHT_VALS.size(), 0.0);
            for (size_t j = 0; j < WEIGHT_VALS.size(); ++j) {
                std::vector<double> wMat = wann::setWeights(ind.wVec, WEIGHT_VALS[j]);
                int correct = 0;
                for (int k = 0; k < N_TASK_POINTS; ++k) {
                    std::vector<double> out = wann::act(wMat, ind.aVec, ind.nInput, ind.nOutput,
                                                         {task.xs[k], task.ys[k]});
                    int pred = out[0] > 0.0 ? 1 : 0;
                    if (pred == task.labels[k]) ++correct;
                }
                reward[i][j] = static_cast<double>(correct) / static_cast<double>(N_TASK_POINTS);
            }
        }
        return reward;
    }

    // -------------------------------------------------------------------
    // Champion layout: layered node-link diagram, keyed by persistent node
    // IDs (bias/input/output IDs are shared across the whole population and
    // never change; only hidden-node IDs are unique-per-mutation) so the
    // pop-in animation can tell "new" from "already existed" across
    // generations even when the champion jumps between lineages.
    // -------------------------------------------------------------------
    struct VisNode { int id; int type; float depth; };
    struct VisEdge { int srcId; int dstId; bool enabled; bool excitatory; };

    struct ChampionLayout {
        std::vector<VisNode> nodes;
        std::vector<VisEdge> edges;
        double bestWeightVal = 1.0;
        int nEnabledConns = 0;
    };

    ChampionLayout computeChampionLayout(const wann::Ind& ind, const std::vector<double>& reward) {
        ChampionLayout layout;

        size_t jBest = 0;
        for (size_t j = 1; j < reward.size(); ++j)
            if (reward[j] > reward[jBest]) jBest = j;
        layout.bestWeightVal = WEIGHT_VALS[jBest];

        auto [Q, wMat] = wann::getNodeOrder(ind.nodes, ind.conns);
        if (Q.empty()) return layout; // cycle guard; shouldn't happen for feed-forward WANN

        const int nNodes  = static_cast<int>(ind.nodes.size());
        const int nIns    = ind.nInput + 1; // bias + inputs
        const int nOut    = ind.nOutput;
        const int nHidden = nNodes - nIns - nOut;

        std::vector<double> hMat(static_cast<size_t>(nHidden) * nHidden, 0.0);
        for (int i = 0; i < nHidden; ++i)
            for (int j = 0; j < nHidden; ++j)
                hMat[i * nHidden + j] = wMat[(nIns + i) * nNodes + (nIns + j)];
        std::vector<double> hLay = (nHidden > 0) ? wann::getLayer(hMat, nHidden) : std::vector<double>{};

        double maxHiddenLayer = 0.0;
        for (double l : hLay) maxHiddenLayer = std::max(maxHiddenLayer, l);
        float outputDepth = (nHidden > 0) ? static_cast<float>(maxHiddenLayer + 2.0) : 1.0f;

        layout.nodes.reserve(nNodes);
        for (int p = 0; p < nNodes; ++p) {
            const auto& nd = ind.nodes[Q[p]];
            float depth;
            if (p < nIns) depth = 0.0f;
            else if (p < nIns + nHidden) depth = static_cast<float>(hLay[p - nIns] + 1.0);
            else depth = outputDepth;
            layout.nodes.push_back({nd.id, nd.type, depth});
        }

        for (int i = 0; i < nNodes; ++i) {
            for (int j = 0; j < nNodes; ++j) {
                double w = wMat[i * nNodes + j];
                bool isNan = std::isnan(w);
                if (!isNan && w == 0.0) continue;
                VisEdge e;
                e.srcId = ind.nodes[Q[i]].id;
                e.dstId = ind.nodes[Q[j]].id;
                e.enabled = !isNan;
                e.excitatory = isNan ? true : (w > 0.0);
                if (e.enabled) ++layout.nEnabledConns;
                layout.edges.push_back(e);
            }
        }
        return layout;
    }
}

int main() {
    g_uiScale = SCREEN_WIDTH / 1680.0f;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Neuroevolución WANN");
    SetTargetFPS(60);

    std::mt19937 seedRng(std::random_device{}());

    wann::Hyperparams hyp;
    hyp.ann_nInput  = 2;
    hyp.ann_nOutput = 1;
    hyp.popSize     = POP_SIZE;
    hyp.maxGen      = 1'000'000;
    hyp.alg_nVals   = static_cast<int>(WEIGHT_VALS.size());
    // Include Squared (11, excluded from the upstream default range) -- it's the natural
    // building block for a radial x^2+y^2 boundary, and cuts stagnation on this task a lot.
    hyp.ann_actRange = {1,2,3,4,5,6,7,8,9,10,11};

    ClassificationTask task;
    wann::Wann* wannAlgo = nullptr;

    // Per-generation display state (rebuilt at generation boundaries, drawn every frame).
    int generation = 0;
    double bestFitness = 0.0, meanFitness = 0.0;
    std::vector<double> popFitness;
    ChampionLayout championLayout;
    std::vector<float> boundaryPixels; // RGBA
    Texture2D boundaryTex{};
    constexpr int BOUNDARY_RES = 96;
    std::deque<float> bestHistory, meanHistory;
    constexpr size_t HISTORY_CAP = 400;

    std::unordered_map<int, double> nodeFirstSeen;
    std::unordered_map<int64_t, double> edgeFirstSeen;
    double solvedHoldTimer = 0.0;

    auto edgeKey = [](int srcId, int dstId) -> int64_t {
        return (static_cast<int64_t>(srcId) << 20) ^ static_cast<int64_t>(dstId);
    };

    auto resetEvolution = [&]() {
        delete wannAlgo;
        wann::seedRng(std::random_device{}());
        task.resample(seedRng);
        wannAlgo = new wann::Wann(hyp);
        generation = 0;
        bestFitness = meanFitness = 0.0;
        popFitness.clear();
        championLayout = ChampionLayout{};
        bestHistory.clear();
        meanHistory.clear();
        nodeFirstSeen.clear();
        edgeFirstSeen.clear();
        solvedHoldTimer = 0.0;
    };

    auto advanceGeneration = [&]() {
        auto& pop = wannAlgo->ask();
        auto reward = evalPopulation(pop, task);
        wannAlgo->tell(reward);
        generation = wannAlgo->generation();

        popFitness.resize(pop.size());
        size_t championIdx = 0;
        double sumFit = 0.0;
        for (size_t i = 0; i < pop.size(); ++i) {
            popFitness[i] = pop[i].fitness;
            sumFit += pop[i].fitness;
            if (pop[i].fitness > pop[championIdx].fitness) championIdx = i;
        }
        bestFitness = pop[championIdx].fitness;
        meanFitness = sumFit / static_cast<double>(pop.size());

        bestHistory.push_back(static_cast<float>(bestFitness));
        meanHistory.push_back(static_cast<float>(meanFitness));
        if (bestHistory.size() > HISTORY_CAP) { bestHistory.pop_front(); meanHistory.pop_front(); }

        championLayout = computeChampionLayout(pop[championIdx], reward[championIdx]);

        double now = GetTime();
        for (const auto& n : championLayout.nodes)
            nodeFirstSeen.try_emplace(n.id, now);
        for (const auto& e : championLayout.edges)
            if (e.enabled) edgeFirstSeen.try_emplace(edgeKey(e.srcId, e.dstId), now);

        // Decision-boundary heatmap for the champion at its best shared weight.
        const wann::Ind& champ = pop[championIdx];
        for (int py = 0; py < BOUNDARY_RES; ++py) {
            double y = 1.0 - 2.0 * (py + 0.5) / BOUNDARY_RES;
            for (int px = 0; px < BOUNDARY_RES; ++px) {
                double x = -1.0 + 2.0 * (px + 0.5) / BOUNDARY_RES;
                double out = evalNetOutput(champ, championLayout.bestWeightVal, x, y);
                double conf = std::clamp(std::abs(out) / 2.0, 0.0, 1.0);
                Color c = lerpColor(WHITE, out > 0.0 ? COLOR_CLASS1 : COLOR_CLASS0, static_cast<float>(0.25 + 0.75 * conf));
                size_t idx = (static_cast<size_t>(py) * BOUNDARY_RES + px) * 4;
                boundaryPixels[idx + 0] = c.r / 255.0f;
                boundaryPixels[idx + 1] = c.g / 255.0f;
                boundaryPixels[idx + 2] = c.b / 255.0f;
                boundaryPixels[idx + 3] = 1.0f;
            }
        }
        std::vector<unsigned char> pixelsU8(boundaryPixels.size());
        for (size_t i = 0; i < boundaryPixels.size(); ++i)
            pixelsU8[i] = static_cast<unsigned char>(std::clamp(boundaryPixels[i], 0.0f, 1.0f) * 255.0f);
        UpdateTexture(boundaryTex, pixelsU8.data());
    };

    // Texture must be created after InitWindow (needs a GL context).
    {
        Image img = GenImageColor(BOUNDARY_RES, BOUNDARY_RES, WHITE);
        boundaryTex = LoadTextureFromImage(img);
        UnloadImage(img);
        SetTextureFilter(boundaryTex, TEXTURE_FILTER_BILINEAR);
        boundaryPixels.assign(static_cast<size_t>(BOUNDARY_RES) * BOUNDARY_RES * 4, 1.0f);
    }

    resetEvolution();
    advanceGeneration(); // show generation 0 immediately instead of a blank first frame

    // --- Layout: 2x2 panel grid (network / boundary / population / chart), title bar on
    // top, controls on the bottom. Sized for 1920x1080 -- a normal single landscape screen,
    // not the dual-4K portrait kiosk the other demos in this repo target.
    constexpr float MARGIN     = 28.0f;
    constexpr float GAP        = 18.0f;
    constexpr float TOP_BAR_H  = 76.0f;
    constexpr float CONTROLS_H = 74.0f;

    // Bottom clearance is BrandingFooterHeight() (not the plain MARGIN used on the other
    // three sides) so the controls row clears DrawSponsorLogos/DrawFondecytCredit -- the
    // flat 28px MARGIN doesn't any more now that both got bigger.
    const float bottomMargin  = BrandingFooterHeight();
    const float contentTop    = MARGIN + TOP_BAR_H + GAP;
    const float contentBottom = SCREEN_HEIGHT - bottomMargin - CONTROLS_H - GAP;
    const float contentLeft   = MARGIN;
    const float contentWidth  = SCREEN_WIDTH - 2.0f * MARGIN;
    const float contentHeight = contentBottom - contentTop;
    const float cellW = (contentWidth - GAP) / 2.0f;
    const float cellH = (contentHeight - GAP) / 2.0f;

    Rectangle networkPanel  = { contentLeft, contentTop, cellW, cellH };
    Rectangle boundaryPanel = { contentLeft + cellW + GAP, contentTop, cellW, cellH };
    Rectangle popPanel      = { contentLeft, contentTop + cellH + GAP, cellW, cellH };
    Rectangle chartPanel    = { contentLeft + cellW + GAP, contentTop + cellH + GAP, cellW, cellH };

    constexpr float BUTTON_WIDTH = 150.0f;
    constexpr float SLIDER_WIDTH = 380.0f;
    constexpr float CONTROL_GAP = 46.0f;
    float controlsWidth = BUTTON_WIDTH + CONTROL_GAP + SLIDER_WIDTH + CONTROL_GAP + BUTTON_WIDTH;
    float controlsLeft = (SCREEN_WIDTH - controlsWidth) / 2.0f;
    float controlsTop = SCREEN_HEIGHT - bottomMargin - CONTROLS_H;

    Button playPauseButton({ controlsLeft, controlsTop + 20.0f, BUTTON_WIDTH, 44.0f }, "Pausar");
    Button resetButton({ controlsLeft + BUTTON_WIDTH + CONTROL_GAP + SLIDER_WIDTH + CONTROL_GAP, controlsTop + 20.0f, BUTTON_WIDTH, 44.0f }, "Reiniciar");

    struct Slider {
        Rectangle track; float minValue, maxValue, value; bool dragging = false;
        void update(Vector2 mouse) {
            Rectangle hit = { track.x - 12, track.y - 12, track.width + 24, track.height + 24 };
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, hit)) dragging = true;
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragging = false;
            if (dragging) {
                float t = std::clamp((mouse.x - track.x) / track.width, 0.0f, 1.0f);
                value = minValue + t * (maxValue - minValue);
            }
        }
        void draw() const {
            float trackY = track.y + track.height / 2.0f;
            DrawLineEx({ track.x, trackY }, { track.x + track.width, trackY }, 4.0f, LIGHTGRAY);
            float t = (value - minValue) / (maxValue - minValue);
            float handleX = track.x + t * track.width;
            DrawLineEx({ track.x, trackY }, { handleX, trackY }, 4.0f, Color{60, 110, 220, 255});
            DrawCircleV({ handleX, trackY }, 10.0f, Color{60, 110, 220, 255});
            DrawCircleLines(static_cast<int>(handleX), static_cast<int>(trackY), 10.0f, DARKGRAY);
        }
    };
    Slider speedSlider{
        { controlsLeft + BUTTON_WIDTH + CONTROL_GAP, controlsTop + 42.0f, SLIDER_WIDTH, 8.0f },
        static_cast<float>(MIN_GENS_PER_SEC), static_cast<float>(MAX_GENS_PER_SEC), static_cast<float>(DEFAULT_GENS_PER_SEC)
    };

    bool running = true;
    double accumulatorSec = 0.0;

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        double dt = GetFrameTime();

        if (playPauseButton.isClicked(mouse)) {
            running = !running;
            playPauseButton.setText(running ? "Pausar" : "Reanudar");
        }
        if (resetButton.isClicked(mouse)) resetEvolution();
        speedSlider.update(mouse);

        bool solved = bestFitness >= AUTO_RESET_FITNESS && generation >= AUTO_RESET_MIN_GEN;
        if (running && !solved) {
            accumulatorSec += dt * speedSlider.value;
            accumulatorSec = std::min(accumulatorSec, 1.0 / speedSlider.value * 20.0);
            while (accumulatorSec >= 1.0 / speedSlider.value) {
                advanceGeneration();
                accumulatorSec -= 1.0 / speedSlider.value;
                solved = bestFitness >= AUTO_RESET_FITNESS && generation >= AUTO_RESET_MIN_GEN;
                if (solved || generation >= AUTO_RESET_MAX_GEN) break;
            }
        }
        if (solved) {
            solvedHoldTimer += dt;
            if (solvedHoldTimer >= SOLVED_HOLD_SECONDS) resetEvolution();
        } else if (generation >= AUTO_RESET_MAX_GEN) {
            solvedHoldTimer += dt;
            if (solvedHoldTimer >= SOLVED_HOLD_SECONDS) resetEvolution();
        } else {
            solvedHoldTimer = 0.0;
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        // --- Title / stats bar ---
        const char* title = "Neuroevolución WANN: aprendiendo a clasificar";
        DrawText(title, static_cast<int>(MARGIN), static_cast<int>(MARGIN - 4), FS(24), DARKGRAY);
        DrawText(TextFormat("Gen %d", generation), static_cast<int>(MARGIN), static_cast<int>(MARGIN + 30), FS(18), DARKGRAY);
        DrawText(TextFormat("Mejor: %.0f%%", bestFitness * 100.0), static_cast<int>(MARGIN + 140 * g_uiScale), static_cast<int>(MARGIN + 30), FS(18), DARKGRAY);
        DrawText(TextFormat("Promedio: %.0f%%", meanFitness * 100.0), static_cast<int>(MARGIN + 300 * g_uiScale), static_cast<int>(MARGIN + 30), FS(18), DARKGRAY);
        DrawText(TextFormat("Conexiones (campeón): %d", championLayout.nEnabledConns), static_cast<int>(MARGIN + 520 * g_uiScale), static_cast<int>(MARGIN + 30), FS(18), DARKGRAY);
        if (solved) {
            const char* msg = "Población resuelta -- reiniciando...";
            int mw = MeasureText(msg, FS(20));
            DrawText(msg, SCREEN_WIDTH - static_cast<int>(MARGIN) - mw, static_cast<int>(MARGIN + 12), FS(20), Color{34, 140, 60, 255});
        }

        // ================= Panel: champion network =================
        DrawRectangleRec(networkPanel, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(networkPanel, 1.5f, LIGHTGRAY);
        DrawText("Red del campeón (mutaciones nuevas destellan)", static_cast<int>(networkPanel.x + 10), static_cast<int>(networkPanel.y + 6), FS(15), DARKGRAY);

        {
            Rectangle plot = { networkPanel.x + 20, networkPanel.y + 30, networkPanel.width - 40, networkPanel.height - 66 };
            float maxDepth = 1.0f;
            for (const auto& n : championLayout.nodes) maxDepth = std::max(maxDepth, n.depth);

            // Group nodes by depth to assign vertical slots.
            std::unordered_map<int, std::vector<const VisNode*>> byDepth;
            for (const auto& n : championLayout.nodes) byDepth[static_cast<int>(std::lround(n.depth))].push_back(&n);

            std::unordered_map<int, Vector2> posById;
            double now = GetTime();
            for (auto& [depthInt, nodesAtDepth] : byDepth) {
                float x = plot.x + (maxDepth > 0 ? (depthInt / maxDepth) : 0.0f) * plot.width;
                int count = static_cast<int>(nodesAtDepth.size());
                for (int i = 0; i < count; ++i) {
                    float y = plot.y + (count == 1 ? plot.height / 2.0f : (i / static_cast<float>(count - 1)) * plot.height);
                    posById[nodesAtDepth[i]->id] = { x, y };
                }
            }

            // Edges first (under nodes). Disabled = faint dashed grey; enabled = coloured, pop-in.
            for (const auto& e : championLayout.edges) {
                auto itS = posById.find(e.srcId), itD = posById.find(e.dstId);
                if (itS == posById.end() || itD == posById.end()) continue;
                if (!e.enabled) {
                    DrawLineEx(itS->second, itD->second, 1.0f * g_uiScale, Fade(GRAY, 0.25f));
                    continue;
                }
                double firstSeen = edgeFirstSeen.count(edgeKey(e.srcId, e.dstId)) ? edgeFirstSeen[edgeKey(e.srcId, e.dstId)] : now;
                float t = smoothstep01((now - firstSeen) / POP_IN_SECONDS);
                Color base = e.excitatory ? COLOR_EXCITATORY : COLOR_INHIBITORY;
                DrawLineEx(itS->second, itD->second, (1.3f + 0.8f * (1.0f - t)) * g_uiScale, Fade(base, 0.35f + 0.4f * t));
            }

            // Nodes.
            for (const auto& n : championLayout.nodes) {
                auto it = posById.find(n.id);
                if (it == posById.end()) continue;
                double firstSeen = nodeFirstSeen.count(n.id) ? nodeFirstSeen[n.id] : now;
                float t = smoothstep01((now - firstSeen) / POP_IN_SECONDS);
                Color base = (n.type == 4) ? COLOR_BIAS : (n.type == 1) ? COLOR_INPUT
                           : (n.type == 2) ? COLOR_OUTPUT : COLOR_HIDDEN;
                float baseRadius = ((n.type == 3) ? 7.0f : 9.0f) * g_uiScale;
                float radius = baseRadius * (0.4f + 0.6f * t);
                if (t < 1.0f) DrawCircleV(it->second, radius + 5.0f * g_uiScale, Fade(WHITE, (1.0f - t) * 0.7f));
                DrawCircleV(it->second, radius, base);
                DrawCircleLines(static_cast<int>(it->second.x), static_cast<int>(it->second.y), radius, Fade(BLACK, 0.4f));
            }

            // Legend.
            float lx = plot.x, ly = plot.y + plot.height + 10.0f;
            auto legendDot = [&](Color c, const char* label) {
                DrawCircleV({ lx + 6.0f, ly + 6.0f }, 6.0f * g_uiScale, c);
                DrawText(label, static_cast<int>(lx + 16), static_cast<int>(ly - 3), FS(12), DARKGRAY);
                lx += static_cast<float>(MeasureText(label, FS(12))) + 30.0f;
            };
            legendDot(COLOR_BIAS, "Sesgo");
            legendDot(COLOR_INPUT, "Entrada");
            legendDot(COLOR_HIDDEN, "Oculta");
            legendDot(COLOR_OUTPUT, "Salida");
            DrawLineEx({ lx, ly + 6.0f }, { lx + 20.0f, ly + 6.0f }, 2.5f * g_uiScale, COLOR_EXCITATORY);
            DrawText("Excitatoria", static_cast<int>(lx + 26), static_cast<int>(ly - 3), FS(12), DARKGRAY);
            lx += 26.0f + MeasureText("Excitatoria", FS(12)) + 24.0f;
            DrawLineEx({ lx, ly + 6.0f }, { lx + 20.0f, ly + 6.0f }, 2.5f * g_uiScale, COLOR_INHIBITORY);
            DrawText("Inhibitoria", static_cast<int>(lx + 26), static_cast<int>(ly - 3), FS(12), DARKGRAY);
        }

        // ================= Panel: decision boundary =================
        DrawRectangleRec(boundaryPanel, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(boundaryPanel, 1.5f, LIGHTGRAY);
        DrawText("Frontera de decisión (fondo = predicción, puntos = etiqueta real)",
                 static_cast<int>(boundaryPanel.x + 10), static_cast<int>(boundaryPanel.y + 6), FS(15), DARKGRAY);
        {
            float side = std::min(boundaryPanel.width - 40.0f, boundaryPanel.height - 56.0f);
            Rectangle plot = { boundaryPanel.x + (boundaryPanel.width - side) / 2.0f, boundaryPanel.y + 34.0f, side, side };
            DrawTexturePro(boundaryTex, { 0, 0, static_cast<float>(BOUNDARY_RES), static_cast<float>(BOUNDARY_RES) },
                            plot, { 0, 0 }, 0.0f, WHITE);
            DrawRectangleLinesEx(plot, 2.0f, DARKGRAY);
            for (int i = 0; i < N_TASK_POINTS; ++i) {
                float px = plot.x + static_cast<float>((task.xs[i] + 1.0) / 2.0) * plot.width;
                float py = plot.y + static_cast<float>((1.0 - task.ys[i]) / 2.0) * plot.height;
                Color c = task.labels[i] == 1 ? COLOR_CLASS1 : COLOR_CLASS0;
                DrawCircleV({ px, py }, 3.2f * g_uiScale, c);
                DrawCircleLines(static_cast<int>(px), static_cast<int>(py), 3.2f * g_uiScale, Fade(BLACK, 0.6f));
            }
        }

        // ================= Panel: population grid =================
        DrawRectangleRec(popPanel, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(popPanel, 1.5f, LIGHTGRAY);
        DrawText("Población completa (color = fitness de cada individuo)",
                 static_cast<int>(popPanel.x + 10), static_cast<int>(popPanel.y + 6), FS(15), DARKGRAY);
        {
            Rectangle plot = { popPanel.x + 18, popPanel.y + 30, popPanel.width - 36, popPanel.height - 62 };
            int n = static_cast<int>(popFitness.size());
            int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(n) * plot.width / std::max(plot.height, 1.0f))));
            cols = std::max(cols, 1);
            int rows = (n + cols - 1) / cols;
            float cellW2 = plot.width / cols, cellH2 = plot.height / rows;
            float boxSize = std::min(cellW2, cellH2) * 0.78f;
            for (int i = 0; i < n; ++i) {
                int col = i % cols, row = i / cols;
                float cx = plot.x + col * cellW2 + cellW2 / 2.0f;
                float cy = plot.y + row * cellH2 + cellH2 / 2.0f;
                Rectangle box = { cx - boxSize / 2.0f, cy - boxSize / 2.0f, boxSize, boxSize };
                DrawRectangleRec(box, fitnessColor(popFitness[static_cast<size_t>(i)]));
                DrawRectangleLinesEx(box, 1.0f, Fade(BLACK, 0.25f));
            }

            // Colour scale legend.
            float lx = plot.x, ly = plot.y + plot.height + 10.0f;
            DrawText("Fitness:", static_cast<int>(lx), static_cast<int>(ly), FS(13), DARKGRAY);
            lx += 66.0f;
            constexpr int SCALE_STEPS = 40;
            float scaleW = 200.0f;
            for (int i = 0; i < SCALE_STEPS; ++i) {
                Color c = fitnessColor(static_cast<double>(i) / (SCALE_STEPS - 1));
                DrawRectangleRec({ lx + i * (scaleW / SCALE_STEPS), ly, scaleW / SCALE_STEPS + 1, 14.0f }, c);
            }
            DrawRectangleLinesEx({ lx, ly, scaleW, 14.0f }, 1.0f, DARKGRAY);
            DrawText("0%", static_cast<int>(lx + scaleW + 8), static_cast<int>(ly - 1), FS(12), GRAY);
            const char* hundred = "100%";
            DrawText(hundred, static_cast<int>(lx - MeasureText(hundred, FS(12)) - 8), static_cast<int>(ly - 1), FS(12), GRAY);
        }

        // ================= Panel: fitness history =================
        DrawRectangleRec(chartPanel, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(chartPanel, 1.5f, LIGHTGRAY);
        DrawText("Fitness a lo largo de las generaciones", static_cast<int>(chartPanel.x + 10), static_cast<int>(chartPanel.y + 6), FS(15), DARKGRAY);
        {
            Rectangle plot = { chartPanel.x + 34, chartPanel.y + 28, chartPanel.width - 54, chartPanel.height - 46 };
            for (int g = 0; g <= 4; ++g) {
                float y = plot.y + plot.height - g / 4.0f * plot.height;
                DrawLineEx({ plot.x, y }, { plot.x + plot.width, y }, 1.0f, Fade(LIGHTGRAY, 0.6f));
                DrawText(TextFormat("%d%%", g * 25), static_cast<int>(plot.x - 32), static_cast<int>(y - 7), FS(11), GRAY);
            }
            auto drawSeries = [&](const std::deque<float>& series, Color c) {
                if (series.size() < 2) return;
                float xStep = plot.width / static_cast<float>(HISTORY_CAP - 1);
                float xOffset = plot.width - (static_cast<float>(series.size()) - 1) * xStep;
                Vector2 prev{};
                for (size_t i = 0; i < series.size(); ++i) {
                    Vector2 p = { plot.x + xOffset + static_cast<float>(i) * xStep, plot.y + plot.height - series[i] * plot.height };
                    if (i > 0) DrawLineEx(prev, p, 2.0f * g_uiScale, c);
                    prev = p;
                }
            };
            drawSeries(meanHistory, Color{140, 140, 150, 255});
            drawSeries(bestHistory, Color{34, 140, 60, 255});

            float lx = plot.x + plot.width - 190.0f, ly = plot.y - 2.0f;
            DrawLineEx({ lx, ly + 6 }, { lx + 20, ly + 6 }, 2.5f, Color{34, 140, 60, 255});
            DrawText("Mejor", static_cast<int>(lx + 26), static_cast<int>(ly - 2), FS(12), DARKGRAY);
            DrawLineEx({ lx + 96, ly + 6 }, { lx + 116, ly + 6 }, 2.5f, Color{140, 140, 150, 255});
            DrawText("Promedio", static_cast<int>(lx + 122), static_cast<int>(ly - 2), FS(12), DARKGRAY);
        }

        // --- Controls ---
        playPauseButton.draw(mouse);
        speedSlider.draw();
        DrawText(TextFormat("Velocidad: %.0f gen/s", speedSlider.value),
                 static_cast<int>(speedSlider.track.x), static_cast<int>(speedSlider.track.y - 22), FS(16), DARKGRAY);
        resetButton.draw(mouse);

        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    delete wannAlgo;
    CloseWindow();
    return 0;
}
