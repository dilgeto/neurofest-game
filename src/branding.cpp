#include <raylib.h>
#include "../include/branding.hpp"

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

    constexpr float LOGO_HEIGHT = 60.0f; // on-screen height each logo is scaled to
    constexpr float LOGO_GAP = 12.0f;    // horizontal gap between logos
    constexpr float MARGIN = 16.0f;      // distance from the screen's bottom-right corner
}

void DrawSponsorLogos(float screenWidth, float screenHeight) {
    float x = screenWidth - MARGIN;
    float y = screenHeight - MARGIN - LOGO_HEIGHT;

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

        float scale = LOGO_HEIGHT / static_cast<float>(logo.texture.height);
        float width = logo.texture.width * scale;
        x -= width;
        DrawTextureEx(logo.texture, { x, y }, 0.0f, scale, WHITE);
        x -= LOGO_GAP;
    }
}
