#pragma once
#include "demo_module.hpp"
#include "button.hpp"
#include "snn_network.hpp"

#include <algorithm>
#include <deque>
#include <functional>
#include <string>
#include <vector>

// Extracted from main_synapse_demo.cpp: shows how a spike travels from a presynaptic neuron
// to a postsynaptic one, and how that changes the postsynaptic membrane voltage, for one
// excitatory pair and one inhibitory pair. Same Izhikevich integration and conductance-based
// synapse mechanics as the main app's SnnNetwork engine. See the original file's header
// comment for the full explanation; unchanged here.
class SynapseModule : public IDemoModule {
public:
    SynapseModule();

    const char* name() const override { return "Transporte de spikes"; }
    void setBounds(Rectangle bounds) override;
    void update(Vector2 mouse, float frameMs) override;
    void draw(Vector2 mouse) const override;

private:
    static constexpr double V_THRESHOLD = 30.0;
    static constexpr double PLOT_V_MIN = -130.0;  // wide enough for the inhibitory pair's deep IPSPs
    static constexpr double PLOT_V_MAX = 40.0;

    struct Neuron {
        SnnNeuronParams params;
        double drivingCurrent;
        double v, u;
        double gExc = 0.0, gInh = 0.0;
        bool spiked = false;
        float flashMs = 0.0f;
        std::deque<float> history;

        Neuron(SnnNeuronType type, double driving);
        void integrate();
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

    // Draws the shared axes/threshold-line chrome for a voltage plot and hands the caller
    // the plot's inner Rectangle plus a value->screenY mapper to overlay one or more traces
    // on identical axes. Template (so it stays header-only) mirrors the free function this
    // was extracted from in main_synapse_demo.cpp.
    template <typename DrawTraces>
    void drawGraphFrame(Rectangle bounds, const std::string& label, DrawTraces drawTraces) const {
        DrawRectangleRec(bounds, Color{250, 250, 251, 255});
        DrawRectangleLinesEx(bounds, 1.5f, LIGHTGRAY);
        DrawText(label.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 8), fontSize(18), DARKGRAY);

        Rectangle plot = { bounds.x + 10, bounds.y + 32, bounds.width - 20, bounds.height - 42 };

        auto valueToY = [&](double v) {
            double t = (v - PLOT_V_MIN) / (PLOT_V_MAX - PLOT_V_MIN);
            return plot.y + plot.height - static_cast<float>(t) * plot.height;
        };

        float thresholdY = static_cast<float>(valueToY(V_THRESHOLD));
        for (float x = plot.x; x < plot.x + plot.width; x += 10.0f) {
            DrawLineEx({ x, thresholdY }, { std::min(x + 6.0f, plot.x + plot.width), thresholdY }, 1.0f, Fade(RED, 0.35f));
        }

        drawTraces(plot, valueToY);
    }

    static void drawTrace(Rectangle plot, const std::function<float(double)>& valueToY, const Neuron& neuron, Color traceColor);
    static int fontSize(float basePx);
    void drawVoltageGraph(Rectangle bounds, const std::string& label, const Neuron& neuron, Color traceColor) const;
    void drawCombinedGraph(Rectangle bounds, const SynapsePair& pair) const;
    void drawDiagram(Rectangle bounds, const SynapsePair& pair) const;

    Rectangle bounds_{};
    Rectangle diagramBounds_[2]{};
    Rectangle combinedBounds_[2]{};  // merged view
    Rectangle preGraphBounds_[2]{};  // split view
    Rectangle postGraphBounds_[2]{}; // split view

    Button toggleViewButton_;
    bool mergedView_ = false;
    std::vector<SynapsePair> pairs_;
    double accumulatorMs_ = 0.0;
};
