#include "../include/snn_network.hpp"
#include "../include/ui_scale.hpp"
#include <raymath.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {
    constexpr double TAU_EXC = 5.0;
    constexpr double TAU_INH = 10.0;
    constexpr double V_THRESHOLD = 30.0;
    constexpr double INPUT_SPIKE_CURRENT = 100.0;
    constexpr double TTFS_THRESHOLD = 1e-9;
    // BIAS_CURRENT for the "small"/"current"/"large" encoders (RLEncoder path): a constant
    // conductance bump applied to the bias node every tick. Same value (50.0 mA) across
    // every wann-cpp SNN task (SnnAcrobotTask.h, SnnDiscMCTask.h, SnnCarTask.h, ...).
    constexpr double BIAS_CURRENT_SMALL = 50.0;

    // Time-to-first-spike encoder (LINEAR mapping): earlier spike = larger value.
    // Matches snn-simulator's TTFSEncoder with duration = SnnNetwork::SIM_WINDOW_MS.
    double ttfsSpikeTime(double normalizedValue) {
        if (normalizedValue < 0.0 || normalizedValue > 1.0 || normalizedValue < TTFS_THRESHOLD) {
            return -1.0;
        }
        double tMax = SnnNetwork::SIM_WINDOW_MS - 1.0;
        double t = std::round((1.0 - normalizedValue) * tMax);
        return std::clamp(t, 0.0, tMax);
    }

    std::vector<std::string> splitCsv(const std::string& line) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) tokens.push_back(token);
        return tokens;
    }
}

SnnNeuronParams snnParamsFor(SnnNeuronType type) {
    switch (type) {
        case SnnNeuronType::RegularSpiking:         return {0.02, 0.2,  -65.0, 8.0};
        case SnnNeuronType::FastSpiking:            return {0.1,  0.2,  -65.0, 2.0};
        case SnnNeuronType::Chattering:              return {0.02, 0.2,  -50.0, 2.0};
        case SnnNeuronType::LowThresholdSpiking:    return {0.02, 0.25, -65.0, 2.0};
        case SnnNeuronType::IntrinsicallyBursting:  return {0.02, 0.2,  -55.0, 4.0};
        case SnnNeuronType::Resonator:               return {0.1,  0.26, -65.0, 2.0};
    }
    return {0.02, 0.2, -65.0, 8.0};
}

SnnNeuronType snnActivationToType(int activationId) {
    switch (activationId) {
        case 1: return SnnNeuronType::RegularSpiking;
        case 2: return SnnNeuronType::FastSpiking;
        case 3: return SnnNeuronType::Chattering;
        case 4: return SnnNeuronType::LowThresholdSpiking;
        case 5: return SnnNeuronType::IntrinsicallyBursting;
        case 6: return SnnNeuronType::Resonator;
        case 7: return SnnNeuronType::FastSpiking;
        case 8: return SnnNeuronType::RegularSpiking;
        case 9: return SnnNeuronType::RegularSpiking;
        case 10: return SnnNeuronType::Chattering;
        default: return SnnNeuronType::RegularSpiking;
    }
}

// nInput/nOutput/bias-forcing below match the SPECIFIC best-performing model saved for
// each task (see models/README or bootstrap_compare_*.py in wann-cpp): all three winning
// models were trained with a fixed encoder/decoder pair, and nInput already reflects that
// encoder's channel count (the "small" encoder uses 2 input neurons per observation
// variable, not 1). Loading a different .out file trained with another encoder for the
// same task would require adjusting nInput here.

const SnnTaskPreset& snnAcrobotPreset() {
    // Winning run: acrobot_small_first_spike (encoder "small" -> nInput = 6 obs * 2 = 12).
    static const SnnTaskPreset preset{"Acrobot", 12, 3, true, {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, SnnEncoderKind::Small};
    return preset;
}

const SnnTaskPreset& snnMountainCarPreset() {
    // The "Mountain Car" logs are actually the discrete Mountain Car task (executable
    // wann_disc_mc, p/disc_mc_snn.json: nInput=2, nOutput=3) saved under the wrong prefix
    // during training -- confirmed in wann-cpp/bootstrap_compare_mountain_car.py. Winning
    // run: mountain_car_small_first_spike (encoder "small" -> nInput = 2 obs * 2 = 4).
    // Bias is NOT forced to RegularSpiking for this task (SnnDiscMCTask::buildNetwork uses
    // the bias node's own activation gene, same as Racing Car).
    static const SnnTaskPreset preset{"Mountain Car", 4, 3, false, {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}, SnnEncoderKind::Small};
    return preset;
}

const SnnTaskPreset& snnRacingCarPreset() {
    // Winning run: car_ttfs_first_spike (encoder "ttfs" -> nInput = 9 obs, no doubling).
    static const SnnTaskPreset preset{"Racing Car", 9, 2, false, {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}, SnnEncoderKind::Ttfs};
    return preset;
}

const SnnTaskPreset& snnIrisPreset() {
    // Evolved locally by neuroevolution-izhikevich/src/Classification/iris_wann/
    // train_iris_wann.py (encoder "ttfs" -> nInput = 4 raw features, no doubling;
    // nOutput = 3 species). weightVals must match WEIGHT_VALS in that script.
    static const SnnTaskPreset preset{"Iris", 4, 3, true, {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}, SnnEncoderKind::Ttfs};
    return preset;
}

bool SnnNetwork::load(const std::string& outPath, const std::string& wiPath, const SnnTaskPreset& preset, Rectangle bounds) {
    std::ifstream outFile(outPath);
    if (!outFile.is_open()) {
        TraceLog(LOG_WARNING, "SnnNetwork: could not open %s", outPath.c_str());
        return false;
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(outFile, line)) {
        if (line.empty()) continue;
        std::vector<std::string> tokens = splitCsv(line);
        std::vector<double> values;
        values.reserve(tokens.size());
        for (const std::string& tok : tokens) values.push_back(std::stod(tok));
        rows.push_back(std::move(values));
    }
    if (rows.empty()) {
        TraceLog(LOG_WARNING, "SnnNetwork: %s is empty", outPath.c_str());
        return false;
    }

    n = static_cast<int>(rows.size());
    nInput = preset.nInput;
    nOutput = preset.nOutput;
    forceBiasRS = preset.forceRegularSpikingBias;
    encoder = preset.encoder;

    weights.assign(n, std::vector<double>(n, 0.0));
    nodes.assign(n, SnnNode{});

    for (int r = 0; r < n; ++r) {
        const std::vector<double>& row = rows[r];
        for (int c = 0; c < n; ++c) {
            double raw = (c < static_cast<int>(row.size())) ? row[c] : 0.0;
            weights[r][c] = std::isnan(raw) ? 0.0 : raw;
        }
        int activationId = (n < static_cast<int>(row.size())) ? static_cast<int>(std::lround(row[n])) : 1;
        bool isBias = (r == 0);
        SnnNeuronType type = (isBias && forceBiasRS) ? SnnNeuronType::RegularSpiking : snnActivationToType(activationId);
        nodes[r].params = snnParamsFor(type);
    }

    synapses.clear();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j || weights[i][j] == 0.0) continue;
            synapses.push_back({i, j, weights[i][j] > 0.0});
        }
    }

    std::ifstream wiFile(wiPath);
    int wi = 0;
    if (wiFile.is_open()) wiFile >> wi;
    wi = std::clamp(wi, 0, 5);
    sharedWeight = preset.weightVals[wi];

    layoutNodes(bounds);
    resetNeurons();
    spikeRaster.clear();
    pulses.clear();
    simClockMs = SIM_WINDOW_MS;
    return true;
}

void SnnNetwork::layoutNodes(Rectangle bounds) {
    std::vector<int> layer(n, 0);

    int hiddenBegin = nInput + 1;
    int hiddenEnd = n - nOutput; // exclusive
    for (int i = hiddenBegin; i < hiddenEnd; ++i) {
        int maxPred = -1;
        for (int j = 0; j < i; ++j) {
            if (weights[j][i] != 0.0) maxPred = std::max(maxPred, layer[j]);
        }
        layer[i] = (maxPred < 0) ? 1 : maxPred + 1;
    }

    int maxHiddenLayer = 0;
    for (int i = hiddenBegin; i < hiddenEnd; ++i) maxHiddenLayer = std::max(maxHiddenLayer, layer[i]);
    for (int i = hiddenEnd; i < n; ++i) layer[i] = maxHiddenLayer + 1;

    int numLayers = maxHiddenLayer + 2; // input column + hidden columns + output column
    float xStep = bounds.width / static_cast<float>(numLayers + 1);

    std::vector<std::vector<int>> nodesByLayer(numLayers);
    for (int i = 0; i < n; ++i) {
        nodes[i].layer = layer[i];
        nodesByLayer[layer[i]].push_back(i);
    }

    for (int l = 0; l < numLayers; ++l) {
        const std::vector<int>& col = nodesByLayer[l];
        float yStep = bounds.height / static_cast<float>(col.size() + 1);
        for (size_t k = 0; k < col.size(); ++k) {
            nodes[col[k]].position = {
                bounds.x + xStep * (l + 1),
                bounds.y + yStep * (static_cast<float>(k) + 1)
            };
        }
    }
}

void SnnNetwork::resetNeurons() {
    for (SnnNode& node : nodes) {
        node.v = node.params.c;
        node.u = node.params.b * node.params.c;
        node.gExc = 0.0;
        node.gInh = 0.0;
        node.spiked = false;
    }
}

void SnnNetwork::propagateAndIntegrate() {
    for (const SnnSynapse& s : synapses) {
        if (!nodes[s.from].spiked) continue;
        if (s.excitatory) nodes[s.to].gExc += sharedWeight;
        else nodes[s.to].gInh += sharedWeight;
    }

    constexpr double dt = 1.0;
    for (SnnNode& node : nodes) {
        node.spiked = false;
        node.gExc *= std::exp(-dt / TAU_EXC);
        node.gInh *= std::exp(-dt / TAU_INH);
        for (int sub = 0; sub < 2; ++sub) {
            double I = node.gExc - node.gInh;
            double halfDt = dt / 2.0;
            node.v += (0.04 * node.v * node.v + 5.0 * node.v + 140.0 - node.u + I) * halfDt;
            node.u += node.params.a * (node.params.b * node.v - node.u) * halfDt;
        }
        if (node.v >= V_THRESHOLD) {
            node.spiked = true;
            node.v = node.params.c;
            node.u += node.params.d;
        }
    }
}

void SnnNetwork::tickTtfs(int t, const std::vector<double>& spikeTimeForNode) {
    for (int i = 0; i < n; ++i) {
        if (spikeTimeForNode[i] == static_cast<double>(t)) {
            nodes[i].gExc += INPUT_SPIKE_CURRENT;
        }
    }
    propagateAndIntegrate();
}

void SnnNetwork::tickSmall(const std::vector<double>& constantCurrentForNode) {
    // RLEncoder::SMALL currents are re-applied every tick (Network::setInputCurrents),
    // unlike the TTFS encoder's one-shot spike-time kick.
    for (int i = 0; i < n; ++i) {
        nodes[i].gExc += constantCurrentForNode[i];
    }
    propagateAndIntegrate();
}

void SnnNetwork::simulateStep(const std::vector<double>& observation) {
    resetNeurons();
    spikeRaster.assign(SIM_WINDOW_MS, std::vector<bool>(n, false));

    if (encoder == SnnEncoderKind::Ttfs) {
        std::vector<double> spikeTimeForNode(n, -1.0);
        spikeTimeForNode[0] = 0.0; // bias: normalized value is always 1.0 -> fires at t=0
        for (int k = 0; k < nInput && (1 + k) < n; ++k) {
            double v = std::clamp(observation[static_cast<size_t>(k)], 0.0, 1.0);
            spikeTimeForNode[1 + k] = ttfsSpikeTime(v);
        }
        for (int t = 0; t < SIM_WINDOW_MS; ++t) {
            tickTtfs(t, spikeTimeForNode);
            for (int i = 0; i < n; ++i) spikeRaster[static_cast<size_t>(t)][i] = nodes[i].spiked;
        }
    } else {
        // RLEncoder::SMALL: 2 input neurons per raw observation variable (negative,
        // positive) -- value<0 drives the negative neuron, value>=0 the positive one,
        // magnitude = |value| (weight=1.0). Bias gets a constant current, not a spike.
        std::vector<double> constantCurrentForNode(n, 0.0);
        constantCurrentForNode[0] = BIAS_CURRENT_SMALL;
        int rawVars = nInput / 2;
        for (int k = 0; k < rawVars; ++k) {
            // RLEncoder::encodeSmall does not clamp or normalize: an unbounded raw value
            // (e.g. an angular velocity of several rad/s) drives a proportionally large
            // current, exactly as it did during training.
            double value = observation[static_cast<size_t>(k)];
            double magnitude = std::abs(value);
            int negIdx = 1 + 2 * k;
            int posIdx = negIdx + 1;
            if (negIdx < n) constantCurrentForNode[negIdx] = (value < 0.0) ? magnitude : 0.0;
            if (posIdx < n) constantCurrentForNode[posIdx] = (value >= 0.0) ? magnitude : 0.0;
        }
        for (int t = 0; t < SIM_WINDOW_MS; ++t) {
            tickSmall(constantCurrentForNode);
            for (int i = 0; i < n; ++i) spikeRaster[static_cast<size_t>(t)][i] = nodes[i].spiked;
        }
    }

    pulses.clear();
    for (int t = 0; t < SIM_WINDOW_MS; ++t) {
        for (int i = 0; i < n; ++i) {
            if (!spikeRaster[static_cast<size_t>(t)][i]) continue;
            for (const SnnSynapse& s : synapses) {
                if (s.from != i) continue;
                pulses.push_back({i, s.to, static_cast<float>(t), static_cast<float>(t + 1), s.excitatory});
            }
        }
    }

    simClockMs = 0.0;
}

void SnnNetwork::advance(double dtMs) {
    simClockMs = std::min(simClockMs + dtMs, static_cast<double>(SIM_WINDOW_MS));
}

int SnnNetwork::decodeFirstSpikeWinner() const {
    int winner = nOutput / 2;
    int earliest = SIM_WINDOW_MS + 1;
    int outputBegin = n - nOutput;
    for (int o = 0; o < nOutput; ++o) {
        int nodeIndex = outputBegin + o;
        for (int t = 0; t < static_cast<int>(spikeRaster.size()); ++t) {
            if (spikeRaster[static_cast<size_t>(t)][nodeIndex]) {
                if (t < earliest) {
                    earliest = t;
                    winner = o;
                }
                break;
            }
        }
    }
    return winner;
}

double SnnNetwork::firstSpikeTimeForOutput(int outputIndex) const {
    int nodeIndex = n - nOutput + outputIndex;
    for (int t = 0; t < static_cast<int>(spikeRaster.size()); ++t) {
        if (spikeRaster[static_cast<size_t>(t)][nodeIndex]) return static_cast<double>(t);
    }
    return -1.0;
}

void SnnNetwork::setIoDisplay(const std::vector<SnnIoEntry>& inputs, const std::vector<SnnIoEntry>& outputs,
                               int highlightedOutput) {
    ioInputs = inputs;
    ioOutputs = outputs;
    ioHighlightedOutput = highlightedOutput;
}

void SnnNetwork::draw(Rectangle bounds) const {
    for (const SnnSynapse& s : synapses) {
        Color base = s.excitatory ? Color{60, 200, 120, 255} : Color{220, 70, 70, 255};
        DrawLineEx(nodes[s.from].position, nodes[s.to].position, 1.5f * g_uiScale, Fade(base, 0.25f));
    }

    for (const Pulse& p : pulses) {
        if (simClockMs < p.startMs || simClockMs > p.endMs) continue;
        float alpha = (p.endMs > p.startMs) ? static_cast<float>((simClockMs - p.startMs) / (p.endMs - p.startMs)) : 0.0f;
        Vector2 pos = Vector2Lerp(nodes[p.from].position, nodes[p.to].position, alpha);
        Color col = p.excitatory ? Color{120, 255, 160, 255} : Color{255, 110, 110, 255};
        DrawCircleV(pos, 4.0f * g_uiScale, col);
    }

    constexpr float FLASH_DURATION_MS = 4.0f;
    for (int i = 0; i < n; ++i) {
        Color base = (i == 0) ? GRAY : (i <= nInput) ? Color{80, 140, 230, 255}
                    : (i >= n - nOutput) ? Color{240, 170, 60, 255}
                    : Color{170, 120, 220, 255};

        float flash = 0.0f;
        if (!spikeRaster.empty()) {
            int lastTick = static_cast<int>(std::floor(simClockMs));
            int firstTick = std::max(0, lastTick - static_cast<int>(FLASH_DURATION_MS));
            for (int t = firstTick; t <= lastTick && t < static_cast<int>(spikeRaster.size()); ++t) {
                if (spikeRaster[static_cast<size_t>(t)][i]) {
                    float age = static_cast<float>(simClockMs - t);
                    flash = std::max(flash, 1.0f - age / FLASH_DURATION_MS);
                }
            }
        }

        float radius = (10.0f + 6.0f * flash) * g_uiScale;
        if (flash > 0.0f) DrawCircleV(nodes[i].position, radius + 4.0f * g_uiScale, Fade(WHITE, flash * 0.5f));
        DrawCircleV(nodes[i].position, radius, base);
        DrawCircleLines(static_cast<int>(nodes[i].position.x), static_cast<int>(nodes[i].position.y), radius, Fade(BLACK, 0.4f));
    }

    const int IO_FONT = static_cast<int>(std::lround(14.0f * g_uiScale));
    const float IO_LINE_GAP = 16.0f * g_uiScale;

    // Input readouts: anchored just left of the node(s) each semantic variable drives.
    // The "small" encoder splits one variable into a negative/positive node pair, so the
    // label/value sit at the midpoint between the two.
    for (size_t k = 0; k < ioInputs.size(); ++k) {
        Vector2 anchor;
        if (encoder == SnnEncoderKind::Small) {
            int negIdx = 1 + 2 * static_cast<int>(k), posIdx = negIdx + 1;
            if (posIdx < n) anchor = Vector2Lerp(nodes[negIdx].position, nodes[posIdx].position, 0.5f);
            else if (negIdx < n) anchor = nodes[negIdx].position;
            else continue;
        } else {
            int idx = 1 + static_cast<int>(k);
            if (idx >= n) continue;
            anchor = nodes[idx].position;
        }

        const SnnIoEntry& entry = ioInputs[k];
        int labelWidth = MeasureText(entry.label.c_str(), IO_FONT);
        int valueWidth = MeasureText(entry.value.c_str(), IO_FONT);
        int textWidth = std::max(labelWidth, valueWidth);
        float textRight = std::max(bounds.x + textWidth, anchor.x - 18.0f * g_uiScale);
        DrawText(entry.label.c_str(), static_cast<int>(textRight - labelWidth), static_cast<int>(anchor.y - IO_LINE_GAP), IO_FONT, DARKGRAY);
        DrawText(entry.value.c_str(), static_cast<int>(textRight - valueWidth), static_cast<int>(anchor.y + 2.0f), IO_FONT, BLACK);
    }

    // Output readouts: anchored just above each output node (stacked label then value); the
    // decoder's current winner (if any) gets a highlighted ring so it's obvious which action
    // is "leaving".
    const float OUTPUT_VALUE_OFFSET = 34.0f * g_uiScale;
    for (size_t o = 0; o < ioOutputs.size(); ++o) {
        int idx = n - nOutput + static_cast<int>(o);
        if (idx < 0 || idx >= n) continue;
        Vector2 anchor = nodes[idx].position;
        bool highlighted = (ioHighlightedOutput == static_cast<int>(o));
        if (highlighted) DrawCircleLines(static_cast<int>(anchor.x), static_cast<int>(anchor.y), 20.0f * g_uiScale, Color{240, 170, 60, 255});

        const SnnIoEntry& entry = ioOutputs[o];
        int labelWidth = MeasureText(entry.label.c_str(), IO_FONT);
        int valueWidth = MeasureText(entry.value.c_str(), IO_FONT);
        float valueY = anchor.y - OUTPUT_VALUE_OFFSET;
        float labelY = std::max(bounds.y + 2.0f, valueY - IO_LINE_GAP);
        float labelX = std::clamp(anchor.x - labelWidth / 2.0f, bounds.x, bounds.x + bounds.width - labelWidth);
        float valueX = std::clamp(anchor.x - valueWidth / 2.0f, bounds.x, bounds.x + bounds.width - valueWidth);
        DrawText(entry.label.c_str(), static_cast<int>(labelX), static_cast<int>(labelY), IO_FONT, DARKGRAY);
        DrawText(entry.value.c_str(), static_cast<int>(valueX), static_cast<int>(valueY),
                 IO_FONT, highlighted ? Color{200, 120, 20, 255} : BLACK);
    }

    // Legend: synapse color key + what each output neuron means, anchored bottom-right.
    if (!ioOutputs.empty()) {
        const int LEGEND_FONT = static_cast<int>(std::lround(15.0f * g_uiScale));
        const float LINE_H = 20.0f * g_uiScale;
        const float PAD = 10.0f * g_uiScale;
        const float SWATCH = 16.0f * g_uiScale;

        int legendLines = 2 + static_cast<int>(ioOutputs.size());
        float legendHeight = PAD * 2.0f + legendLines * LINE_H;
        float legendWidth = 280.0f * g_uiScale;
        Rectangle legendBox = { bounds.x + bounds.width - legendWidth, bounds.y + bounds.height - legendHeight, legendWidth, legendHeight };
        DrawRectangleRounded(legendBox, 0.1f, 6, Fade(RAYWHITE, 0.88f));
        DrawRectangleRoundedLinesEx(legendBox, 0.1f, 6, 1.0f * g_uiScale, Fade(GRAY, 0.6f));

        float lx = legendBox.x + PAD;
        float ly = legendBox.y + PAD;
        DrawLineEx({ lx, ly + SWATCH / 2.0f }, { lx + SWATCH, ly + SWATCH / 2.0f }, 3.0f * g_uiScale, Color{60, 200, 120, 255});
        DrawText("Sinapsis excitatoria", static_cast<int>(lx + SWATCH + 8.0f * g_uiScale), static_cast<int>(ly), LEGEND_FONT, DARKGRAY);
        ly += LINE_H;
        DrawLineEx({ lx, ly + SWATCH / 2.0f }, { lx + SWATCH, ly + SWATCH / 2.0f }, 3.0f * g_uiScale, Color{220, 70, 70, 255});
        DrawText("Sinapsis inhibitoria", static_cast<int>(lx + SWATCH + 8.0f * g_uiScale), static_cast<int>(ly), LEGEND_FONT, DARKGRAY);
        ly += LINE_H;

        for (size_t o = 0; o < ioOutputs.size(); ++o) {
            bool highlighted = (ioHighlightedOutput == static_cast<int>(o));
            Color dot = highlighted ? Color{240, 170, 60, 255} : Fade(Color{240, 170, 60, 255}, 0.5f);
            DrawCircleV({ lx + SWATCH / 2.0f, ly + SWATCH / 2.0f }, SWATCH / 2.0f, dot);
            std::string text = "Salida " + std::to_string(o) + ": " + ioOutputs[o].label;
            DrawText(text.c_str(), static_cast<int>(lx + SWATCH + 8.0f * g_uiScale), static_cast<int>(ly), LEGEND_FONT, DARKGRAY);
            ly += LINE_H;
        }
    }
}
