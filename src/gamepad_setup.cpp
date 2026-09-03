#include "../include/gamepad_setup.hpp"
#include <raylib.h>

// raylib/GLFW only reads gamepad axes/buttons through glfwGetGamepadState(), which requires
// the controller's exact USB VID:PID to be present in GLFW's built-in SDL_GameControllerDB
// snapshot -- otherwise IsGamepadAvailable() is true but every axis/button silently reads 0.
// The GameSir Nova Lite's 2.4GHz dongle (and its wired USB-C mode, which enumerates
// identically) shows up as a generic "Zikway HID gamepad" (bus/vendor/product/version
// 0003:3537:1041:0111), which isn't in that snapshot. This mapping was reverse-engineered by
// reading raw events off /dev/input/js0 while pressing one physical control at a time (RIGHT
// trigger alone -> raw axis 4, LEFT trigger alone -> raw axis 5; the two face buttons chosen
// for A/B -> raw buttons 0/1). The right stick is still unmapped since nothing reads it.
//
// The d-pad needed a different treatment than /dev/input/js0 suggests: that legacy "joydev"
// API flattens a hat switch into two synthetic extra axes (6 and 7 here), but GLFW's actual
// Linux backend reads the modern evdev API directly and keeps a hat as a genuine hat,
// separate from axisCount -- confirmed via /sys/class/input/eventN/device/capabilities/abs,
// which shows this device has only 6 real axes (0-5, the ones already mapped below) plus one
// hat. Referencing "a6"/"a7" (out of range) made GLFW discard this *entire* mapping as
// invalid, not just the d-pad -- hence "dpup:h0.1" etc. below (hat 0, SDL bitmask: 1=up,
// 2=right, 4=down, 8=left) instead. A different GameSir unit/firmware revision (or a
// different controller entirely) will need all of this re-derived the same way.
void SetupGameSirGamepadMapping() {
    SetGamepadMappings(
        "03000000373500004110000011010000,GameSir Nova Lite (dongle),"
        "platform:Linux,leftx:a0,lefty:a1,lefttrigger:a5,righttrigger:a4,"
        "a:b0,b:b1,dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,"
    );
}
