#pragma once
#include <raylib.h>
#include <string>
#include <vector>

// Reproduces the Izhikevich SNN simulator from wann-cpp/snn-simulator closely enough to
// animate real spike propagation: same neuron model, same encoders (TTFS and "small"),
// same one-tick synaptic delay. Input values are synthetic (not read from a recorded
// episode), so no per-task observation-normalization formulas are needed here.

enum class SnnNeuronType {
    RegularSpiking,
    FastSpiking,
    Chattering,
    LowThresholdSpiking,
    IntrinsicallyBursting,
    Resonator
};

struct SnnNeuronParams {
    double a, b, c, d;
};

SnnNeuronParams snnParamsFor(SnnNeuronType type);
SnnNeuronType snnActivationToType(int activationId);

// Which encoder wann-cpp used to turn a raw observation vector into input-neuron drive:
//   Ttfs  - one input neuron per observation channel, each emits at most one spike whose
//           timing encodes the (already-normalized-to-[0,1]) value (earlier = larger).
//   Small - two input neurons per RAW observation variable (negative/positive), each
//           driven by a constant current for the whole window (RLEncoder::SMALL).
enum class SnnEncoderKind { Ttfs, Small };

// Fixed per-task constants pulled from wann-cpp (nInput/nOutput, whether the bias node's
// neuron type is hardcoded to RegularSpiking by that task's buildNetwork(), and the
// WEIGHT_VALS table that ".wi" indexes into).
struct SnnTaskPreset {
    std::string name;
    int nInput;
    int nOutput;
    bool forceRegularSpikingBias;
    double weightVals[6];
    SnnEncoderKind encoder = SnnEncoderKind::Ttfs;
};

const SnnTaskPreset& snnAcrobotPreset();
const SnnTaskPreset& snnMountainCarPreset();
const SnnTaskPreset& snnRacingCarPreset();

struct SnnNode {
    SnnNeuronParams params;
    double v = 0.0;
    double u = 0.0;
    double gExc = 0.0;
    double gInh = 0.0;
    bool spiked = false;
    int layer = 0;
    Vector2 position{};
};

struct SnnSynapse {
    int from;
    int to;
    bool excitatory;
};

// One 20ms simulated window per call to simulateStep(); advance() scrubs playback through
// the pre-computed spike raster so drawing/timing stays decoupled from the Izhikevich math.
class SnnNetwork {
    public:
        static constexpr int SIM_WINDOW_MS = 20;

        // Parses a WANN-exported network: `outPath` is the NxN(+1) weight/activation matrix
        // (Ind::exportNet format), `wiPath` is the single shared-weight index.
        bool load(const std::string& outPath, const std::string& wiPath, const SnnTaskPreset& preset, Rectangle bounds);

        // Runs the network's encoder + Izhikevich simulation for one 20ms window given an
        // observation vector of size syntheticInputSize():
        //   Ttfs  -> one value per input neuron, in [0,1] (spike timing).
        //   Small -> one RAW value per observation variable (nInput/2) -- e.g. cos/sin
        //            components and unclamped angular velocities -- sign/magnitude become
        //            constant drive on two input neurons each (RLEncoder::encodeSmall does
        //            not clamp or normalize, so neither do we).
        void simulateStep(const std::vector<double>& observation);

        SnnEncoderKind encoderKind() const { return encoder; }

        // Number of values simulateStep() expects, and (for synthetic/random testing) the
        // natural range ([0,1] for Ttfs, [-1,1] for Small) each one should be drawn from.
        int syntheticInputSize() const { return (encoder == SnnEncoderKind::Small) ? nInput / 2 : nInput; }

        // First-spike decoder (RLDecoder::FIRST_SPIKE / SnnAcrobotTask.cpp's discrete
        // branch): among the nOutput output neurons, the one whose first spike in the just-
        // simulated window came earliest wins; defaults to nOutput/2 if none spiked.
        int decodeFirstSpikeWinner() const;

        // Tick (0..SIM_WINDOW_MS-1) of output neuron `outputIndex`'s first spike in the
        // just-simulated window, or -1 if it never spiked. Used for RLDecoder's continuous
        // (per-output) decoding, e.g. Racing Car's (throttle, steering) pair.
        double firstSpikeTimeForOutput(int outputIndex) const;

        int outputCount() const { return nOutput; }

        // Scrubs the playback cursor forward by `dtMs` of simulated time.
        void advance(double dtMs);

        bool isStepFinished() const { return simClockMs >= SIM_WINDOW_MS; }

        void draw(Rectangle bounds) const;

        int nodeCount() const { return static_cast<int>(nodes.size()); }
        int inputCount() const { return nInput; }

    private:
        struct Pulse {
            int from, to;
            float startMs, endMs;
            bool excitatory;
        };

        void layoutNodes(Rectangle bounds);
        void resetNeurons();
        void propagateAndIntegrate();
        void tickTtfs(int t, const std::vector<double>& spikeTimeForNode);
        void tickSmall(const std::vector<double>& constantCurrentForNode);

        int n = 0;
        int nInput = 0;
        int nOutput = 0;
        bool forceBiasRS = false;
        double sharedWeight = 1.0;
        SnnEncoderKind encoder = SnnEncoderKind::Ttfs;

        std::vector<std::vector<double>> weights; // NxN, 0 = no synapse, sign = polarity
        std::vector<SnnNode> nodes;
        std::vector<SnnSynapse> synapses;

        std::vector<std::vector<bool>> spikeRaster; // [tick][node]
        std::vector<Pulse> pulses;
        double simClockMs = SIM_WINDOW_MS;
};
