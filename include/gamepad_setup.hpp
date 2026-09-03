#pragma once

// Registers the GameSir Nova Lite's raylib/GLFW gamepad mapping (see the .cpp for the full
// reverse-engineering story). Call once, right after InitWindow(), in every binary that
// reads gamepad axes/buttons (NeuroGame, MultiDemo) -- forgetting this in one of them is a
// silent failure: IsGamepadAvailable() still reports true, but every axis/button reads 0.
void SetupGameSirGamepadMapping();
