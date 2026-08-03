#include "../include/button.hpp"

/**
 *
 * 
 * @param 
 */
Button::Button(Rectangle button, string text) : button(button), text(text) {}

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