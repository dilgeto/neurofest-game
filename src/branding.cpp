#include <raylib.h>
#include <cmath>
#include "../include/branding.hpp"
#include "../include/ui_scale.hpp"

namespace {
    struct SponsorLogo {
        const char* path;
        Texture2D texture{};
        bool loadAttempted = false;
    };

    // Left-to-right on screen: anid, diinf, fondecyt.
    SponsorLogo sponsorLogos[] = {
        { "img/anid.jpeg" },
        { "img/diinf.png" },
        { "img/fondecyt.png" },
    };
    constexpr int SPONSOR_LOGO_COUNT = 3;

    // Base (unscaled) sizes -- multiplied by g_uiScale at draw time below, since g_uiScale
    // is only set once main() knows SCREEN_WIDTH.
    constexpr float BASE_LOGO_HEIGHT = 60.0f; // on-screen height each logo is scaled to
    constexpr float BASE_LOGO_GAP = 12.0f;    // horizontal gap between logos
    constexpr float BASE_MARGIN = 16.0f;      // distance from the screen's bottom-right corner
}

void DrawSponsorLogos(float screenWidth, float screenHeight) {
    float logoHeight = BASE_LOGO_HEIGHT * g_uiScale;
    float logoGap = BASE_LOGO_GAP * g_uiScale;
    float margin = BASE_MARGIN * g_uiScale;

    float x = screenWidth - margin;
    float y = screenHeight - margin - logoHeight;

    // Walk right-to-left so each logo's on-screen width (only known once its texture is
    // loaded) can be subtracted from the running cursor before placing the next one to its
    // left -- this naturally reproduces the anid/diinf/fondecyt left-to-right order above.
    for (int i = SPONSOR_LOGO_COUNT - 1; i >= 0; --i) {
        SponsorLogo& logo = sponsorLogos[i];
        if (!logo.loadAttempted) {
            logo.texture = LoadTexture(logo.path);
            logo.loadAttempted = true;
        }
        if (logo.texture.id == 0) continue; // failed to load; skip without shifting others

        float scale = logoHeight / static_cast<float>(logo.texture.height);
        float width = logo.texture.width * scale;
        x -= width;
        DrawTextureEx(logo.texture, { x, y }, 0.0f, scale, WHITE);
        x -= logoGap;
    }
}

void DrawFondecytCredit(float screenWidth, float screenHeight) {
    (void)screenWidth;
    int font = static_cast<int>(std::lround(14.0f * g_uiScale));
    float margin = 16.0f * g_uiScale;
    float lineGap = 16.0f * g_uiScale;
    constexpr const char* LINES[] = {
        "Fondecyt Regular 1251455",
        "NeuroMetaEvo: Integrating Metaheuristic Techniques with Neuroscience",
        "for Advanced Neuromorphic Algorithm Design",
    };
    constexpr int LINE_COUNT = 3;

    float y = screenHeight - margin - LINE_COUNT * lineGap;
    for (int i = 0; i < LINE_COUNT; ++i) {
        DrawText(LINES[i], static_cast<int>(margin), static_cast<int>(y + i * lineGap), font, Fade(DARKGRAY, 0.8f));
    }
}
