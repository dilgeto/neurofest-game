#pragma once
#include <raylib.h>

#include "button.hpp"

using namespace std;

class Menu {
    private:
        Button vsAI;
        Button buildNetwork;
        Button playNetwork;
        Button quit;

    public:
        Menu();
};