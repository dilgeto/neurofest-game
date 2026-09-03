#include "../include/synapse_module.hpp"
#include "../include/ui_scale.hpp"
#include <raymath.h>

#include <algorithm>
#include <cmath>

namespace {
    constexpr double SIM_DT_MS = 0.5;         // Izhikevich integration step
    constexpr double SIM_SPEED_FACTOR = 0.1;  // slow real biological time 10x so individual
                                               // spikes/EPSPs/IPSPs are easy to follow
    constexpr double TAU_EXC = 5.0;
    constexpr double TAU_INH = 10.0;

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
}

int SynapseModule::fontSize(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }

SynapseModule::Neuron::Neuron(SnnNeuronType type, double driving)
    : params(snnParamsFor(type)), drivingCurrent(driving), v(params.c), u(params.b * params.c) {
    history.assign(HISTORY_CAPACITY, static_cast<float>(v));
}

void SynapseModule::Neuron::integrate() {
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

void SynapseModule::drawTrace(Rectangle plot, const std::function<float(double)>& valueToY, const Neuron& neuron, Color traceColor) {
    float xStep = plot.width / static_cast<float>(HISTORY_CAPACITY - 1);
    Vector2 prev{};
    for (size_t i = 0; i < neuron.history.size(); ++i) {
        Vector2 p = { plot.x + xStep * static_cast<float>(i), valueToY(neuron.history[i]) };
        if (i > 0) DrawLineEx(prev, p, 2.0f, traceColor);
        prev = p;
    }
}

void SynapseModule::drawVoltageGraph(Rectangle bounds, const std::string& label, const Neuron& neuron, Color traceColor) const {
    drawGraphFrame(bounds, label, [&](Rectangle plot, const auto& valueToY) {
        drawTrace(plot, valueToY, neuron, traceColor);
    });
}

// Both traces on one set of axes, e.g. the toggled "merged" view -- pre and post overlaid so
// their causal relationship (a presynaptic spike's effect on the postsynaptic trace) is
// visible directly, instead of split across two panels.
void SynapseModule::drawCombinedGraph(Rectangle bounds, const SynapsePair& pair) const {
    drawGraphFrame(bounds, "Presináptica + Postsináptica", [&](Rectangle plot, const auto& valueToY) {
        drawTrace(plot, valueToY, pair.pre, PRE_COLOR);
        drawTrace(plot, valueToY, pair.post, pair.excitatory ? EXC_POST_COLOR : INH_POST_COLOR);
    });

    DrawText("Pre", static_cast<int>(bounds.x + bounds.width - 90), static_cast<int>(bounds.y + 8), fontSize(16), PRE_COLOR);
    DrawText("Post", static_cast<int>(bounds.x + bounds.width - 50), static_cast<int>(bounds.y + 8), fontSize(16),
        pair.excitatory ? EXC_POST_COLOR : INH_POST_COLOR);
}

void SynapseModule::drawDiagram(Rectangle bounds, const SynapsePair& pair) const {
    DrawRectangleRec(bounds, Color{250, 250, 251, 255});
    DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
    DrawText(pair.title.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), fontSize(18), DARKGRAY);

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

    DrawText("Pre", static_cast<int>(preCenter.x - 15), static_cast<int>(preCenter.y + 26), fontSize(16), GRAY);
    DrawText("Post", static_cast<int>(postCenter.x - 17), static_cast<int>(postCenter.y + 26), fontSize(16), GRAY);
}

SynapseModule::SynapseModule() : toggleViewButton_({0, 0, 1, 1}, "Vista: Gráficas separadas") {
    pairs_.push_back({
        "Sinapsis excitatoria", true, EXC_WEIGHT, Color{60, 170, 90, 255},
        Neuron(SnnNeuronType::RegularSpiking, PRE_DRIVING_CURRENT),
        Neuron(SnnNeuronType::RegularSpiking, EXC_POST_DRIVING_CURRENT),
        {}
    });
    pairs_.push_back({
        "Sinapsis inhibitoria", false, INH_WEIGHT, Color{220, 90, 70, 255},
        Neuron(SnnNeuronType::RegularSpiking, PRE_DRIVING_CURRENT),
        Neuron(SnnNeuronType::RegularSpiking, INH_POST_DRIVING_CURRENT),
        {}
    });
}

// rowHeight is already fully derived from bounds.height (fills whatever's available), so --
// like IzhikevichModule and unlike DecoderModule's fixed-pixel budget -- no fraction-of-a-
// reference-height treatment is needed here, just the bounds_ offset/substitution.
void SynapseModule::setBounds(Rectangle bounds) {
    bounds_ = bounds;
    const Rectangle& b = bounds;

    constexpr float MARGIN_X = 40.0f;
    constexpr float MARGIN_TOP = 90.0f;
    constexpr float MARGIN_BOTTOM = 30.0f;
    constexpr float GAP = 20.0f;
    constexpr float DIAGRAM_WIDTH = 360.0f;
    float graphWidth = (b.width - 2.0f * MARGIN_X - DIAGRAM_WIDTH - 2.0f * GAP) / 2.0f;
    float combinedGraphWidth = graphWidth * 2.0f + GAP;
    float rowHeight = (b.height - MARGIN_TOP - MARGIN_BOTTOM - GAP) / 2.0f;

    toggleViewButton_.setBounds({ b.x + (b.width - 300.0f) / 2.0f, b.y + 20.0f, 300.0f, 50.0f });

    for (int row = 0; row < 2; ++row) {
        float y = b.y + MARGIN_TOP + row * (rowHeight + GAP);
        diagramBounds_[row] = { b.x + MARGIN_X, y, DIAGRAM_WIDTH, rowHeight };
        combinedBounds_[row] = { b.x + MARGIN_X + DIAGRAM_WIDTH + GAP, y, combinedGraphWidth, rowHeight };
        preGraphBounds_[row] = { b.x + MARGIN_X + DIAGRAM_WIDTH + GAP, y, graphWidth, rowHeight };
        postGraphBounds_[row] = { preGraphBounds_[row].x + graphWidth + GAP, y, graphWidth, rowHeight };
    }
}

void SynapseModule::update(Vector2 mouse, float frameMs) {
    if (toggleViewButton_.isClicked(mouse)) {
        mergedView_ = !mergedView_;
        toggleViewButton_.setText(mergedView_ ? "Vista: Gráfica fusionada" : "Vista: Gráficas separadas");
    }

    for (SynapsePair& pair : pairs_) {
        for (float& p : pair.pulses) p += frameMs;
        pair.pulses.erase(
            std::remove_if(pair.pulses.begin(), pair.pulses.end(), [](float p) { return p >= PULSE_DURATION_MS; }),
            pair.pulses.end());
        pair.pre.flashMs = std::max(0.0f, pair.pre.flashMs - frameMs);
        pair.post.flashMs = std::max(0.0f, pair.post.flashMs - frameMs);
    }

    accumulatorMs_ += static_cast<double>(frameMs) * SIM_SPEED_FACTOR;
    accumulatorMs_ = std::min(accumulatorMs_, SIM_DT_MS * 400.0); // cap catch-up on a stall
    while (accumulatorMs_ >= SIM_DT_MS) {
        for (SynapsePair& pair : pairs_) {
            if (pair.pre.spiked) {
                if (pair.excitatory) pair.post.gExc += pair.weight;
                else pair.post.gInh += pair.weight;
                pair.pulses.push_back(0.0f);
            }
            pair.pre.integrate();
            pair.post.integrate();
        }
        accumulatorMs_ -= SIM_DT_MS;
    }
}

void SynapseModule::draw(Vector2 mouse) const {
    toggleViewButton_.draw(mouse);

    for (int row = 0; row < static_cast<int>(pairs_.size()); ++row) {
        const SynapsePair& pair = pairs_[static_cast<size_t>(row)];
        drawDiagram(diagramBounds_[row], pair);

        if (mergedView_) {
            drawCombinedGraph(combinedBounds_[row], pair);
        } else {
            drawVoltageGraph(preGraphBounds_[row], "Presináptica", pair.pre, PRE_COLOR);
            drawVoltageGraph(postGraphBounds_[row], "Postsináptica", pair.post,
                pair.excitatory ? EXC_POST_COLOR : INH_POST_COLOR);
        }
    }
}
