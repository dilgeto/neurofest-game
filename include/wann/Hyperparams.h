#pragma once
#include <string>
#include <vector>

namespace wann {

// Trimmed copy of wann-cpp's Hyperparams.h: same field set, minus the JSON
// file-loading helpers (loadHyp/updateHyp) so this demo doesn't need to pull
// in nlohmann/json just to hardcode a handful of overrides in code.
struct Hyperparams {
    // --- algorithm ---
    std::string task          = "swingup";
    std::string alg_wDist     = "standard";
    int    alg_nVals          = 6;
    int    alg_nReps          = 4;
    double alg_probMoo        = 0.80;
    int    maxGen             = 2048;
    int    popSize            = 128;

    // --- mutation probabilities ---
    double prob_crossover          = 0.0;
    double prob_mutAct             = 0.50;
    double prob_addNode            = 0.25;
    double prob_addConn            = 0.20;
    double prob_enable             = 0.05;
    double prob_initEnable         = 0.5;
    double prob_toggleExcitatory   = 0.10;

    // --- selection ---
    double select_cullRatio   = 0.2;
    double select_eliteRatio  = 0.2;
    int    select_tournSize   = 8;

    // --- I/O ---
    int    save_mod           = 8;
    int    bestReps           = 20;

    // --- task-specific ---
    int    ann_nInput         = 5;
    int    ann_nOutput        = 1;
    int    ann_initAct        = 1;
    std::vector<int> ann_actRange = {1,2,3,4,5,6,7,8,9,10};
    double ann_absWCap        = 2.0;

    // --- SNN interface (unused by this demo, kept for struct compatibility) ---
    std::string snn_encoder   = "poisson";
    std::string snn_decoder   = "rate";
    int         snn_neurons_per_var = 5;
    double reward_shaping_scale = 0.0;
    bool snn_reset_between_steps = true;
};

} // namespace wann
