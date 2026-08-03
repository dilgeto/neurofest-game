#include "../include/model_browser.hpp"

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

const std::vector<SnnTaskCategory>& snnTaskCategories() {
    static const std::vector<SnnTaskCategory> categories = {
        {"Acrobot", "models/acrobot", snnAcrobotPreset},
        {"Mountain Car", "models/mountain_car", snnMountainCarPreset},
        {"Racing Car", "models/racing_car", snnRacingCarPreset},
    };
    return categories;
}

std::vector<SnnModelEntry> listSnnModels(const std::string& directory) {
    std::vector<SnnModelEntry> entries;
    if (!fs::exists(directory) || !fs::is_directory(directory)) return entries;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".out") continue;

        fs::path wiPath = entry.path();
        wiPath.replace_extension(".wi");
        if (!fs::exists(wiPath)) continue;

        std::string displayName = entry.path().stem().string();
        std::replace(displayName.begin(), displayName.end(), '_', ' ');

        entries.push_back({displayName, entry.path().string(), wiPath.string()});
    }

    std::sort(entries.begin(), entries.end(), [](const SnnModelEntry& a, const SnnModelEntry& b) {
        return a.displayName < b.displayName;
    });
    return entries;
}
