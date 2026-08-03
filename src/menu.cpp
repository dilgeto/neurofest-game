#include <raylib.h>
#include "../include/menu.hpp"

namespace {
    const float BUTTON_WIDTH = 220;
    const float BUTTON_HEIGHT = 50;
    const float BUTTON_GAP = 10;

    // Lays out button `index` of `total` in a vertical stack centered on (screenWidth/2, centerY)
    Rectangle stackedButton(float screenWidth, float centerY, int index, int total) {
        float totalHeight = total * BUTTON_HEIGHT + (total - 1) * BUTTON_GAP;
        float startY = centerY - totalHeight / 2.0f;
        return {
            screenWidth / 2.0f - BUTTON_WIDTH / 2.0f,
            startY + index * (BUTTON_HEIGHT + BUTTON_GAP),
            BUTTON_WIDTH,
            BUTTON_HEIGHT
        };
    }
}

Menu::Menu(float screenWitdh, float screenHeight) :
    status(MAIN_MENU),
    // Main menu
    vsAI(stackedButton(screenWitdh, screenHeight / 2.0f, 0, 3), "Jugar contra la IA"),
    playNetwork(stackedButton(screenWitdh, screenHeight / 2.0f, 1, 3), "Evaluar red"),
    quit(stackedButton(screenWitdh, screenHeight / 2.0f, 2, 3), "Salir"),
    // VS AI menu (task selection)
    selectAcrobot(stackedButton(screenWitdh, screenHeight / 2.0f, 0, 4), "Acrobot"),
    selectMountainCar(stackedButton(screenWitdh, screenHeight / 2.0f, 1, 4), "Mountain Car"),
    selectRacingCar(stackedButton(screenWitdh, screenHeight / 2.0f, 2, 4), "Racing Car"),
    backFromVsAI(stackedButton(screenWitdh, screenHeight / 2.0f, 3, 4), "Volver"),
    // Play Network menu
    loadNetwork(stackedButton(screenWitdh, screenHeight / 2.0f, 0, 3), "Cargar red existente"),
    createNetwork(stackedButton(screenWitdh, screenHeight / 2.0f, 1, 3), "Crear nueva red"),
    backFromPlayNetwork(stackedButton(screenWitdh, screenHeight / 2.0f, 2, 3), "Volver"),
    // Network view (SNN visualizer)
    backFromNetworkView({20, 20, 140, 40}, "Volver"),
    // Load network: task selection
    loadAcrobotTask(stackedButton(screenWitdh, screenHeight / 2.0f, 0, 4), "Acrobot"),
    loadMountainCarTask(stackedButton(screenWitdh, screenHeight / 2.0f, 1, 4), "Mountain Car"),
    loadRacingCarTask(stackedButton(screenWitdh, screenHeight / 2.0f, 2, 4), "Racing Car"),
    backFromLoadTask(stackedButton(screenWitdh, screenHeight / 2.0f, 3, 4), "Volver"),
    // Load network: file selection
    backFromLoadFile({20, 20, 140, 40}, "Volver")
    {}

void Menu::draw(Vector2 mousePosition) const {
    switch (status) {
        case MAIN_MENU:
            vsAI.draw(mousePosition);
            playNetwork.draw(mousePosition);
            quit.draw(mousePosition);
            break;

        case VS_AI_MENU:
            selectAcrobot.draw(mousePosition);
            selectMountainCar.draw(mousePosition);
            selectRacingCar.draw(mousePosition);
            backFromVsAI.draw(mousePosition);
            break;

        case PLAY_NETWORK_MENU:
            loadNetwork.draw(mousePosition);
            createNetwork.draw(mousePosition);
            backFromPlayNetwork.draw(mousePosition);
            break;

        case NETWORK_VIEW:
            backFromNetworkView.draw(mousePosition);
            break;

        case LOAD_NETWORK_TASK_MENU:
            loadAcrobotTask.draw(mousePosition);
            loadMountainCarTask.draw(mousePosition);
            loadRacingCarTask.draw(mousePosition);
            backFromLoadTask.draw(mousePosition);
            break;

        case LOAD_NETWORK_FILE_MENU:
            backFromLoadFile.draw(mousePosition);
            break;
    }
}
