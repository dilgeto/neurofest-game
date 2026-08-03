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
        // Setters
        void setStatus(int status) {this->status = status;}
        // Drawing
        void draw(Vector2 mousePosition) const;
};