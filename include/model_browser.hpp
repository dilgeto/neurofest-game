#pragma once
#include "snn_network.hpp"
#include <string>
#include <vector>

// A single loadable network found under models/<task>/: a matched .out/.wi pair.
struct SnnModelEntry {
    std::string displayName;
    std::string outPath;
    std::string wiPath;
};

// One entry per task shown in the "Cargar red existente" task-selection screen.
struct SnnTaskCategory {
    std::string label;
    std::string directory; // e.g. "models/acrobot"
    const SnnTaskPreset& (*preset)();
};

const std::vector<SnnTaskCategory>& snnTaskCategories();

// Scans `directory` for `<name>.out` files that have a matching `<name>.wi`, sorted by name.
std::vector<SnnModelEntry> listSnnModels(const std::string& directory);
