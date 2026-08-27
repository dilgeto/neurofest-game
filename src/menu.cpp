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
    selectAcrobot(stackedButton(screenWitdh, screenHeight / 2.0f, 0, 5), "Acrobot"),
    selectMountainCar(stackedButton(screenWitdh, screenHeight / 2.0f, 1, 5), "Mountain Car"),
    selectRacingCar(stackedButton(screenWitdh, screenHeight / 2.0f, 2, 5), "Racing Car"),
    selectIris(stackedButton(screenWitdh, screenHeight / 2.0f, 3, 5), "Iris (Adivina la flor)"),
    backFromVsAI(stackedButton(screenWitdh, screenHeight / 2.0f, 4, 5), "Volver"),
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
    backFromLoadFile({20, 20, 140, 40}, "Volver"),
    // VS AI: Racing Car
    backFromVsAiRacingCar({20, 20, 140, 40}, "Volver"),
    // VS AI: Acrobot / Mountain Car
    backFromVsAiAcrobot({20, 20, 140, 40}, "Volver"),
    backFromVsAiMountainCar({20, 20, 140, 40}, "Volver"),
    // Square, no text -- main.cpp draws a direction arrow on top of each instead.
    leftActionButton({screenWitdh / 2.0f - 110.0f, screenHeight - 130.0f, 100.0f, 100.0f}, ""),
    rightActionButton({screenWitdh / 2.0f + 10.0f, screenHeight - 130.0f, 100.0f, 100.0f}, ""),
    // VS AI: Iris -- 3 species buttons in a centered row; "Siguiente flor" shares the
    // same row (only one of the two groups is ever drawn at a time, by main.cpp, based
    // on round phase). Row y (620) must stay in sync with IRIS_BUTTONS_ROW_Y in main.cpp
    // -- both sit just below irisLeftPanelBounds/irisRightPanelBounds there.
    backFromVsAiIris({20, 20, 140, 40}, "Volver"),
    guessSetosa({screenWitdh / 2.0f - 320.0f, 620.0f, 200.0f, 60.0f}, "Setosa"),
    guessVersicolor({screenWitdh / 2.0f - 100.0f, 620.0f, 200.0f, 60.0f}, "Versicolor"),
    guessVirginica({screenWitdh / 2.0f + 120.0f, 620.0f, 200.0f, 60.0f}, "Virginica"),
    irisNextRound({screenWitdh / 2.0f - 110.0f, 620.0f, 220.0f, 60.0f}, "Siguiente flor")
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
            selectIris.draw(mousePosition);
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

        case VS_AI_RACING_CAR:
            backFromVsAiRacingCar.draw(mousePosition);
            break;

        case VS_AI_ACROBOT:
            backFromVsAiAcrobot.draw(mousePosition);
            leftActionButton.draw(mousePosition);
            rightActionButton.draw(mousePosition);
            break;

        case VS_AI_MOUNTAIN_CAR:
            backFromVsAiMountainCar.draw(mousePosition);
            leftActionButton.draw(mousePosition);
            rightActionButton.draw(mousePosition);
            break;

        case VS_AI_IRIS:
            // Guess/next-round buttons are drawn by main.cpp instead (conditionally, by
            // round phase) -- see the VS_AI_IRIS drawing block in main.cpp.
            backFromVsAiIris.draw(mousePosition);
            break;
    }
}
