#include "../include/button.hpp"

/**
 *
 * 
 * @param 
 */
Button::Button(Rectangle button, string text) : button(button), text(text), fontSize(20) {}

/**
 *
 * 
 * @param 
 */
bool Button::isClicked(Vector2 mousePosition) const {
    return (CheckCollisionPointRec(mousePosition, this->button) && 
        IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) ? true : false;
}

/**
 *
 * 
 * @param 
 */
bool Button::isBeingClicked(Vector2 mousePosition) const {
    return (CheckCollisionPointRec(mousePosition, this->button) && 
        IsMouseButtonDown(MOUSE_BUTTON_LEFT)) ? true : false;
}

void Button::setText(string text) {this->text = text;}
void Button::setFontSize(int fontsize) {this->fontSize = fontsize;}

void Button::draw(Vector2 mousePosition) const {
    bool hovered = CheckCollisionPointRec(mousePosition, button);
    DrawRectangleRec(button, hovered ? LIGHTGRAY : GRAY);
    DrawRectangleLinesEx(button, 2, DARKGRAY);

    int textWidth = MeasureText(text.c_str(), fontSize);
    int textX = static_cast<int>(button.x + (button.width - textWidth) / 2.0f);
    int textY = static_cast<int>(button.y + (button.height - fontSize) / 2.0f);
    DrawText(text.c_str(), textX, textY, fontSize, BLACK);
}