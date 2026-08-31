#include <raylib.h>
#include <raymath.h>
#include <algorithm>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "../include/branding.hpp"
#include "../include/button.hpp"
#include "../include/snn_network.hpp"

// Standalone demo (separate binary from NeuroGame): shows how a spike travels from a
// presynaptic neuron to a postsynaptic one, and how that changes the postsynaptic
// membrane voltage, for one excitatory pair and one inhibitory pair. Same Izhikevich
// integration and conductance-based synapse mechanics as the main app's SnnNetwork engine
// (snn_network.cpp's propagateAndIntegrate: g_exc/g_inh decay each tick, a presynaptic
// spike adds the synapse weight to the postsynaptic conductance one tick later).

namespace {
    constexpr int SCREEN_WIDTH = 1680;
    constexpr int SCREEN_HEIGHT = 900;

    constexpr double SIM_DT_MS = 0.5;         // Izhikevich integration step
    constexpr double SIM_SPEED_FACTOR = 0.1;  // slow real biological time 10x so individual
                                               // spikes/EPSPs/IPSPs are easy to follow
    constexpr double TAU_EXC = 5.0;
    constexpr double TAU_INH = 10.0;
    constexpr double V_THRESHOLD = 30.0;
    constexpr double PLOT_V_MIN = -130.0;     // wide enough for the inhibitory pair's deep IPSPs
    constexpr double PLOT_V_MAX = 40.0;

    constexpr double WINDOW_MS = 220.0;       // sim-time history shown per trace
    constexpr size_t HISTORY_CAPACITY = static_cast<size_t>(WINDOW_MS / SIM_DT_MS);

    constexpr float PULSE_DURATION_MS = 150.0f;  // wall-clock time a pulse dot takes to
                                                  // cross the synapse line (exaggerated for
                                                  // visibility, not the true 1-tick delay)
    constexpr float FLASH_MAX_MS = 120.0f;       // how long a neuron circle stays "lit" after firing

    // Both postsynaptic neurons get zero driving current of their own -- every bit of their
    // membrane response comes from the synapse, not from an artificial baseline. The
    // excitatory pair mixes sub-threshold EPSPs with occasional postsynaptic spikes; the
    // inhibitory pair shows pure IPSPs dipping below rest (it never fires, since inhibition
    // alone can only pull voltage down from an otherwise-quiescent baseline).
    constexpr double PRE_DRIVING_CURRENT = 14.0;
    constexpr double EXC_POST_DRIVING_CURRENT = 0.0;
    constexpr double INH_POST_DRIVING_CURRENT = 0.0;
    constexpr double EXC_WEIGHT = 20.0;
    constexpr double INH_WEIGHT = 80.0;

    const Color PRE_COLOR = Color{60, 110, 220, 255};
    const Color EXC_POST_COLOR = Color{60, 170, 90, 255};
    const Color INH_POST_COLOR = Color{220, 90, 70, 255};

    struct Neuron {
        SnnNeuronParams params;
        double drivingCurrent;
        double v, u;
        double gExc = 0.0, gInh = 0.0;
        bool spiked = false;
        float flashMs = 0.0f;
        std::deque<float> history;

        Neuron(SnnNeuronType type, double driving)
            : params(snnParamsFor(type)), drivingCurrent(driving), v(params.c), u(params.b * params.c) {
            history.assign(HISTORY_CAPACITY, static_cast<float>(v));
        }

        void integrate() {
            spiked = false;
            gExc *= std::exp(-SIM_DT_MS / TAU_EXC);
            gInh *= std::exp(-SIM_DT_MS / TAU_INH);
            for (int sub = 0; sub < 2; ++sub) {
                double halfDt = SIM_DT_MS / 2.0;
                double current = drivingCurrent + gExc - gInh;
                v += (0.04 * v * v + 5.0 * v + 140.0 - u + current) * halfDt;
                u += params.a * (params.b * v - u) * halfDt;
            }
            if (v >= V_THRESHOLD) {
                spiked = true;
                v = params.c;
                u += params.d;
                flashMs = FLASH_MAX_MS;
            }
            history.push_back(static_cast<float>(std::min(v, V_THRESHOLD)));
            if (history.size() > HISTORY_CAPACITY) history.pop_front();
        }
    };

    struct SynapsePair {
        std::string title;
        bool excitatory;
        double weight;
        Color synapseColor;
        Neuron pre;
        Neuron post;
        std::vector<float> pulses; // elapsed wall-clock ms since each in-flight pulse spawned
    };

    // Draws the shared axes/threshold-line chrome for a voltage plot and returns the
    // plot's inner Rectangle plus a value->screenY mapper, so callers can then overlay
    // one or more traces on identical axes.
    template <typename DrawTraces>
    void drawGraphFrame(Rectangle bounds, const std::string& label, DrawTraces drawTraces) {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(label.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), 18, DARKGRAY);

        Rectangle plot = { bounds.x + 10, bounds.y + 32, bounds.width - 20, bounds.height - 42 };

        auto valueToY = [&](double v) {
            double t = (v - PLOT_V_MIN) / (PLOT_V_MAX - PLOT_V_MIN);
            return plot.y + plot.height - static_cast<float>(t) * plot.height;
        };

        float thresholdY = valueToY(V_THRESHOLD);
        for (float x = plot.x; x < plot.x + plot.width; x += 10.0f) {
            DrawLineEx({ x, thresholdY }, { std::min(x + 6.0f, plot.x + plot.width), thresholdY }, 1.0f, Fade(RED, 0.35f));
        }

        drawTraces(plot, valueToY);
    }

    void drawTrace(Rectangle plot, const std::function<float(double)>& valueToY, const Neuron& neuron, Color traceColor) {
        float xStep = plot.width / static_cast<float>(HISTORY_CAPACITY - 1);
        Vector2 prev{};
        for (size_t i = 0; i < neuron.history.size(); ++i) {
            Vector2 p = { plot.x + xStep * static_cast<float>(i), valueToY(neuron.history[i]) };
            if (i > 0) DrawLineEx(prev, p, 2.0f, traceColor);
            prev = p;
        }
    }

    void drawVoltageGraph(Rectangle bounds, const std::string& label, const Neuron& neuron, Color traceColor) {
        drawGraphFrame(bounds, label, [&](Rectangle plot, const auto& valueToY) {
            drawTrace(plot, valueToY, neuron, traceColor);
        });
    }

    // Both traces on one set of axes, e.g. the toggled "merged" view -- pre and post
    // overlaid so their causal relationship (a presynaptic spike's effect on the
    // postsynaptic trace) is visible directly, instead of split across two panels.
    void drawCombinedGraph(Rectangle bounds, const SynapsePair& pair) {
        drawGraphFrame(bounds, "Presinaptica + Postsinaptica", [&](Rectangle plot, const auto& valueToY) {
            drawTrace(plot, valueToY, pair.pre, PRE_COLOR);
            drawTrace(plot, valueToY, pair.post, pair.excitatory ? EXC_POST_COLOR : INH_POST_COLOR);
        });

        DrawText("Pre", static_cast<int>(bounds.x + bounds.width - 90), static_cast<int>(bounds.y + 8), 16, PRE_COLOR);
        DrawText("Post", static_cast<int>(bounds.x + bounds.width - 50), static_cast<int>(bounds.y + 8), 16,
            pair.excitatory ? EXC_POST_COLOR : INH_POST_COLOR);
    }

    void drawDiagram(Rectangle bounds, const SynapsePair& pair) {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(pair.title.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), 18, DARKGRAY);

        Vector2 preCenter = { bounds.x + bounds.width * 0.30f, bounds.y + bounds.height * 0.58f };
        Vector2 postCenter = { bounds.x + bounds.width * 0.70f, bounds.y + bounds.height * 0.58f };

        DrawLineEx(preCenter, postCenter, 3.0f, Fade(pair.synapseColor, 0.55f));
        // Arrow head/tail hint showing signal direction and synapse type (arrow = excitatory,
        // flat bar = inhibitory), same convention as textbook circuit diagrams.
        Vector2 dir = Vector2Normalize(Vector2Subtract(postCenter, preCenter));
        Vector2 perp = { -dir.y, dir.x };
        Vector2 tip = Vector2Subtract(postCenter, Vector2Scale(dir, 22.0f));
        if (pair.excitatory) {
            Vector2 a = Vector2Add(tip, Vector2Scale(perp, 8.0f));
            Vector2 b = Vector2Subtract(tip, Vector2Scale(perp, 8.0f));
            Vector2 c = Vector2Add(tip, Vector2Scale(dir, 14.0f));
            DrawTriangle(b, a, c, pair.synapseColor);
        } else {
            Vector2 a = Vector2Add(tip, Vector2Scale(perp, 10.0f));
            Vector2 b = Vector2Subtract(tip, Vector2Scale(perp, 10.0f));
            DrawLineEx(a, b, 4.0f, pair.synapseColor);
        }

        for (float p : pair.pulses) {
            float t = std::clamp(p / PULSE_DURATION_MS, 0.0f, 1.0f);
            Vector2 pos = Vector2Lerp(preCenter, postCenter, t);
            DrawCircleV(pos, 6.0f, pair.synapseColor);
        }

        float preFlash = std::clamp(pair.pre.flashMs / FLASH_MAX_MS, 0.0f, 1.0f);
        float postFlash = std::clamp(pair.post.flashMs / FLASH_MAX_MS, 0.0f, 1.0f);
        if (preFlash > 0.0f) DrawCircleV(preCenter, 24.0f + 8.0f * preFlash, Fade(WHITE, preFlash * 0.6f));
        if (postFlash > 0.0f) DrawCircleV(postCenter, 24.0f + 8.0f * postFlash, Fade(WHITE, postFlash * 0.6f));
        DrawCircleV(preCenter, 18.0f, PRE_COLOR);
        DrawCircleV(postCenter, 18.0f, pair.excitatory ? EXC_POST_COLOR : INH_POST_COLOR);
        DrawCircleLines(static_cast<int>(preCenter.x), static_cast<int>(preCenter.y), 18.0f, DARKGRAY);
        DrawCircleLines(static_cast<int>(postCenter.x), static_cast<int>(postCenter.y), 18.0f, DARKGRAY);

        DrawText("Pre", static_cast<int>(preCenter.x - 15), static_cast<int>(preCenter.y + 26), 16, GRAY);
        DrawText("Post", static_cast<int>(postCenter.x - 17), static_cast<int>(postCenter.y + 26), 16, GRAY);
    }
}

int main() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Transporte de spikes: sinapsis excitatoria vs inhibitoria");
    SetTargetFPS(60);

    std::vector<SynapsePair> pairs;
    pairs.push_back({
        "Sinapsis excitatoria", true, EXC_WEIGHT, Color{60, 170, 90, 255},
        Neuron(SnnNeuronType::RegularSpiking, PRE_DRIVING_CURRENT),
        Neuron(SnnNeuronType::RegularSpiking, EXC_POST_DRIVING_CURRENT),
        {}
    });
    pairs.push_back({
        "Sinapsis inhibitoria", false, INH_WEIGHT, Color{220, 90, 70, 255},
        Neuron(SnnNeuronType::RegularSpiking, PRE_DRIVING_CURRENT),
        Neuron(SnnNeuronType::RegularSpiking, INH_POST_DRIVING_CURRENT),
        {}
    });

    constexpr float MARGIN_X = 40.0f;
    constexpr float MARGIN_TOP = 90.0f;
    constexpr float MARGIN_BOTTOM = 30.0f;
    constexpr float GAP = 20.0f;
    constexpr float DIAGRAM_WIDTH = 360.0f;
    float graphWidth = (SCREEN_WIDTH - 2.0f * MARGIN_X - DIAGRAM_WIDTH - 2.0f * GAP) / 2.0f;
    float combinedGraphWidth = graphWidth * 2.0f + GAP;
    float rowHeight = (SCREEN_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM - GAP) / 2.0f;

    Button toggleViewButton({ (SCREEN_WIDTH - 300.0f) / 2.0f, 20.0f, 300.0f, 50.0f }, "Vista: Graficas separadas");
    bool mergedView = false;

    double accumulatorMs = 0.0;

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        if (toggleViewButton.isClicked(mouse)) {
            mergedView = !mergedView;
            toggleViewButton.setText(mergedView ? "Vista: Grafica fusionada" : "Vista: Graficas separadas");
        }

        float frameMs = static_cast<float>(GetFrameTime() * 1000.0);

        for (SynapsePair& pair : pairs) {
            for (float& p : pair.pulses) p += frameMs;
            pair.pulses.erase(
                std::remove_if(pair.pulses.begin(), pair.pulses.end(), [](float p) { return p >= PULSE_DURATION_MS; }),
                pair.pulses.end());
            pair.pre.flashMs = std::max(0.0f, pair.pre.flashMs - frameMs);
            pair.post.flashMs = std::max(0.0f, pair.post.flashMs - frameMs);
        }

        accumulatorMs += static_cast<double>(frameMs) * SIM_SPEED_FACTOR;
        accumulatorMs = std::min(accumulatorMs, SIM_DT_MS * 400.0); // cap catch-up on a stall
        while (accumulatorMs >= SIM_DT_MS) {
            for (SynapsePair& pair : pairs) {
                if (pair.pre.spiked) {
                    if (pair.excitatory) pair.post.gExc += pair.weight;
                    else pair.post.gInh += pair.weight;
                    pair.pulses.push_back(0.0f);
                }
                pair.pre.integrate();
                pair.post.integrate();
            }
            accumulatorMs -= SIM_DT_MS;
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        toggleViewButton.draw(mouse);

        for (int row = 0; row < static_cast<int>(pairs.size()); ++row) {
            const SynapsePair& pair = pairs[static_cast<size_t>(row)];
            float y = MARGIN_TOP + row * (rowHeight + GAP);

            Rectangle diagramBounds = { MARGIN_X, y, DIAGRAM_WIDTH, rowHeight };
            drawDiagram(diagramBounds, pair);

            if (mergedView) {
                Rectangle combinedBounds = { MARGIN_X + DIAGRAM_WIDTH + GAP, y, combinedGraphWidth, rowHeight };
                drawCombinedGraph(combinedBounds, pair);
            } else {
                Rectangle preGraphBounds = { MARGIN_X + DIAGRAM_WIDTH + GAP, y, graphWidth, rowHeight };
                Rectangle postGraphBounds = { preGraphBounds.x + graphWidth + GAP, y, graphWidth, rowHeight };
                drawVoltageGraph(preGraphBounds, "Presinaptica", pair.pre, PRE_COLOR);
                drawVoltageGraph(postGraphBounds, "Postsinaptica", pair.post,
                    pair.excitatory ? EXC_POST_COLOR : INH_POST_COLOR);
            }
        }

        DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
        DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
