#include <raylib.h>
#include "../include/menu.hpp"

Menu::Menu() : 
    vsAI(Rectangle , text), 
    buildNetwork(Rectangle , text), 
    playNetwork(Rectangle button, string text),
    quit(Rectangle button, string text)
    {}