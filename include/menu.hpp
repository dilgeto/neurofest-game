#pragma once
#include <raylib.h>

#include "button.hpp"

// Statuses
#define MAIN_MENU 0
#define VS_AI_MENU 1
#define PLAY_NETWORK_MENU 2
#define NETWORK_VIEW 3
#define LOAD_NETWORK_TASK_MENU 4
#define LOAD_NETWORK_FILE_MENU 5
#define VS_AI_RACING_CAR 6
#define VS_AI_ACROBOT 7
#define VS_AI_MOUNTAIN_CAR 8
#define VS_AI_IRIS 9

using namespace std;

class Menu {
    private:
        int status;
        // Main menu
        Button vsAI;
        Button playNetwork;
        Button quit;
        // VS AI menu (task selection)
        Button selectAcrobot;
        Button selectMountainCar;
        Button selectRacingCar;
        Button selectIris;
        Button backFromVsAI;
        // Play Network menu
        Button loadNetwork;
        Button createNetwork;
        Button backFromPlayNetwork;
        // Network view (SNN visualizer)
        Button backFromNetworkView;
        // Load network: task selection
        Button loadAcrobotTask;
        Button loadMountainCarTask;
        Button loadRacingCarTask;
        Button backFromLoadTask;
        // Load network: file selection (list itself is built dynamically outside Menu)
        Button backFromLoadFile;
        // VS AI: Racing Car (human vs AI, no network shown)
        Button backFromVsAiRacingCar;
        // VS AI: Acrobot / Mountain Car (human vs AI, discrete left/right on-screen buttons)
        Button backFromVsAiAcrobot;
        Button backFromVsAiMountainCar;
        Button leftActionButton;
        Button rightActionButton;
        // VS AI: Iris (human vs AI, shared flower classification)
        Button backFromVsAiIris;
        Button guessSetosa;
        Button guessVersicolor;
        Button guessVirginica;
        Button irisNextRound;

    public:
        // Methods
        Menu(float screenWitdh, float screenHeight);
        // Getters
        int getStatus() {return status;}
        Button getVsAI() {return vsAI;}
        Button getPlayNetwork() {return playNetwork;}
        Button getQuit() {return quit;}
        Button getSelectAcrobot() {return selectAcrobot;}
        Button getSelectMountainCar() {return selectMountainCar;}
        Button getSelectRacingCar() {return selectRacingCar;}
        Button getSelectIris() {return selectIris;}
        Button getBackFromVsAI() {return backFromVsAI;}
        Button getLoadNetwork() {return loadNetwork;}
        Button getCreateNetwork() {return createNetwork;}
        Button getBackFromPlayNetwork() {return backFromPlayNetwork;}
        Button getBackFromNetworkView() {return backFromNetworkView;}
        Button getLoadAcrobotTask() {return loadAcrobotTask;}
        Button getLoadMountainCarTask() {return loadMountainCarTask;}
        Button getLoadRacingCarTask() {return loadRacingCarTask;}
        Button getBackFromLoadTask() {return backFromLoadTask;}
        Button getBackFromLoadFile() {return backFromLoadFile;}
        Button getBackFromVsAiRacingCar() {return backFromVsAiRacingCar;}
        Button getBackFromVsAiAcrobot() {return backFromVsAiAcrobot;}
        Button getBackFromVsAiMountainCar() {return backFromVsAiMountainCar;}
        Button getLeftActionButton() {return leftActionButton;}
        Button getRightActionButton() {return rightActionButton;}
        Button getBackFromVsAiIris() {return backFromVsAiIris;}
        Button getGuessSetosa() {return guessSetosa;}
        Button getGuessVersicolor() {return guessVersicolor;}
        Button getGuessVirginica() {return guessVirginica;}
        Button getIrisNextRound() {return irisNextRound;}
        // Setters
        void setStatus(int status) {this->status = status;}
        // Drawing
        void draw(Vector2 mousePosition) const;
};