#pragma once
#include <raylib.h>
#include <string>

using namespace std;

/**/
class Button {
    private:
        Rectangle button;
        string text;
        int fontSize;
    public:
        // Methods
        Button(Rectangle button, string text);
        bool isClicked(Vector2 mousePosition) const;
        bool isBeingClicked(Vector2 mousePosition) const;
        void draw(Vector2 mousePosition) const;

        // Getters
        Rectangle getButton() const {return button;}
        string getText() const {return text;}
        float getPositionX() const {return button.x;}
        float getPositionY() const {return button.y;}
        int getFontSize() const {return fontSize;}
        // Setters
        void setText(string text);
        void setFontSize(int fontsize);
        void setBounds(Rectangle bounds) {button = bounds;}
};