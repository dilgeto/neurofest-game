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
        Rectangle getButton() {return button;}
        string getText() {return text;}
        float getPositionX() {return button.x;}
        float getPositionY() {return button.y;}
        int getFontSize() {return fontSize;}
        // Setters
        void setText(string text);
        void setFontSize(int fontsize);
};