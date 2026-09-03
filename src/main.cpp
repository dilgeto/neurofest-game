#include <raylib.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "../include/acrobot_env.hpp"
#include "../include/branding.hpp"
#include "../include/car_env.hpp"
#include "../include/gamepad_setup.hpp"
#include "../include/iris_env.hpp"
#include "../include/menu.hpp"
#include "../include/model_browser.hpp"
#include "../include/mountain_car_env.hpp"
#include "../include/snn_network.hpp"
#include "../include/ui_scale.hpp"

// Screen dimensions
// Full 3840x2160 monitor, rotated to portrait -- fills it via FLAG_FULLSCREEN_MODE below.
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

// Scales a base (1680-wide-reference) pixel size by g_uiScale, rounding to the nearest
// integer -- use for every DrawText/MeasureText font-size argument and any other fixed pixel
// size (radii, gaps) that should stay visually proportionate as SCREEN_WIDTH/HEIGHT change.
static int FS(float basePx) { return static_cast<int>(std::lround(basePx * g_uiScale)); }

// Playback speed levels, cycled with SPACE: each factor stretches real time by that much so
// individual spike hops become easier to follow, from "Tiempo real" (1x) through "Lenta"
// (20x) to "Ultra lenta" (100x) for tracing single impulses/synapses in fine detail.
static constexpr double SNN_SPEED_FACTORS[] = { 1.0, 20.0, 1000.0 };
static constexpr const char* SNN_SPEED_LABELS[] = { "Tiempo real", "Lenta", "Ultra lenta" };
static constexpr int SNN_SPEED_LEVEL_COUNT = 3;

// Wall-clock duration of one real env step, per task -- what "Real" speed maps the 20ms
// SNN decision window to. Acrobot and Racing Car have an exact physical DT (ported
// straight from rl-tools' environment parameters); Mountain Car's rl-tools step is
// dimensionless (no physical DT), so it falls back to this project's own real-time
// playback convention for it (wann-cpp/replay_mountain_car.py defaults to --fps 30).
static double realStepMsForTask(int taskCategory) {
	switch (taskCategory) {
		case 0: return 200.0;           // Acrobot: DT = 0.2 s (acrobot_env.hpp)
		case 2: return 10.0;            // Racing Car: DT = 0.01 s (car_env.hpp)
		default: return 1000.0 / 30.0;  // Mountain Car: 30 fps convention
	}
}

// Lays out one button per model entry in a vertical, centered stack.
static std::vector<Button> buildModelButtons(const std::vector<SnnModelEntry>& entries) {
	std::vector<Button> buttons;
	const float width = 420.0f;
	const float height = 44.0f;
	const float gap = 8.0f;
	float totalHeight = entries.size() * height + (entries.empty() ? 0.0f : (entries.size() - 1) * gap);
	float startY = SCREEN_HEIGHT / 2.0f - totalHeight / 2.0f;
	for (size_t i = 0; i < entries.size(); ++i) {
		Rectangle rect = {
			SCREEN_WIDTH / 2.0f - width / 2.0f,
			startY + static_cast<float>(i) * (height + gap),
			width,
			height
		};
		buttons.emplace_back(rect, entries[i].displayName);
	}
	return buttons;
}

// Draws a solid triangular direction arrow centered inside `bounds` (used on the square
// izquierda/derecha buttons, which carry no text of their own).
static void drawArrowIcon(Rectangle bounds, bool pointsRight, Color color) {
	float cx = bounds.x + bounds.width / 2.0f;
	float cy = bounds.y + bounds.height / 2.0f;
	float half = std::min(bounds.width, bounds.height) * 0.28f;
	Vector2 tip = { cx + (pointsRight ? half : -half), cy };
	Vector2 baseNear = { cx + (pointsRight ? -half : half), cy - half };
	Vector2 baseFar = { cx + (pointsRight ? -half : half), cy + half };
	// DrawTriangle only fills counter-clockwise-wound triangles; mirroring tip/base for the
	// left-pointing arrow flips the winding, so swap the argument order to compensate.
	if (pointsRight) DrawTriangle(baseNear, baseFar, tip, color);
	else DrawTriangle(baseFar, baseNear, tip, color);
}

// Normalizes a raw CarEnv observation into [0,1] per channel, matching
// SnnCarTask.cpp's TTFS-path normalization exactly (position/heading/velocities scaled
// by their bounds; the 3 lidar readings are already in [0,1]).
static std::vector<double> encodeCarObservation(const std::array<double, 9>& obs) {
	return {
		(obs[0] + CarEnv::BOUND) / (2.0 * CarEnv::BOUND),
		(obs[1] + CarEnv::BOUND) / (2.0 * CarEnv::BOUND),
		(obs[2] + M_PI) / (2.0 * M_PI),
		(obs[3] + CarEnv::VX_MAX) / (2.0 * CarEnv::VX_MAX),
		(obs[4] + CarEnv::VY_MAX) / (2.0 * CarEnv::VY_MAX),
		(obs[5] + CarEnv::OMEGA_MAX) / (2.0 * CarEnv::OMEGA_MAX),
		obs[6], obs[7], obs[8]
	};
}

// RLDecoder::FIRST_SPIKE continuous decode (rlDecoder.cpp's single-arg overload, as used
// by SnnCarTask.cpp's decodeActions): earlier spike -> value closer to +1; no spike -> 0
// before the *2-1 remap, i.e. -1 after it.
static double decodeCarContinuousAction(const SnnNetwork& net, int outputIndex) {
	double t = net.firstSpikeTimeForOutput(outputIndex);
	double value = (t < 0.0) ? 0.0 : (1.0 - t / SnnNetwork::SIM_WINDOW_MS);
	return std::clamp(value * 2.0 - 1.0, -1.0, 1.0);
}

// Feeds SnnNetwork the human-readable input/output readouts for "Evaluar red" (NETWORK_VIEW):
// per-task observation labels/live values on the input side, and per-task action labels on
// the output side, with the currently-applied discrete action (if any) highlighted. Reads
// the same env/pending-action state takeEnvStep() just produced, so labels stay in sync with
// the spike animation currently playing (that window's input drove that window's output).
static void updateSnnIoDisplay(SnnNetwork& net, int taskCategory, const AcrobotEnv& acrobotEnv,
		const MountainCarEnv& mountainCarEnv, const CarEnv& carEnv,
		double pendingAction, double pendingThrottle, double pendingSteering) {
	if (taskCategory == 0) {
		std::array<double, 6> obs = acrobotEnv.observe();
		std::vector<SnnIoEntry> inputs = {
			{"cos(theta1)", TextFormat("%.2f", obs[0])},
			{"sin(theta1)", TextFormat("%.2f", obs[1])},
			{"cos(theta2)", TextFormat("%.2f", obs[2])},
			{"sin(theta2)", TextFormat("%.2f", obs[3])},
			{"Vel. angular 1", TextFormat("%.2f rad/s", obs[4])},
			{"Vel. angular 2", TextFormat("%.2f rad/s", obs[5])},
		};
		int winner = static_cast<int>(std::lround(pendingAction + 1.0));
		std::vector<SnnIoEntry> outputs = {
			{"Torque -1", (winner == 0) ? "ACTIVA" : ""},
			{"Sin torque", (winner == 1) ? "ACTIVA" : ""},
			{"Torque +1", (winner == 2) ? "ACTIVA" : ""},
		};
		net.setIoDisplay(inputs, outputs, winner);
	} else if (taskCategory == 1) {
		std::array<double, 2> obs = mountainCarEnv.observe();
		std::vector<SnnIoEntry> inputs = {
			{"Posición", TextFormat("%.2f", obs[0])},
			{"Velocidad", TextFormat("%.3f", obs[1])},
		};
		int winner = static_cast<int>(std::lround(pendingAction + 1.0));
		std::vector<SnnIoEntry> outputs = {
			{"Empuje izquierda", (winner == 0) ? "ACTIVA" : ""},
			{"Sin empuje", (winner == 1) ? "ACTIVA" : ""},
			{"Empuje derecha", (winner == 2) ? "ACTIVA" : ""},
		};
		net.setIoDisplay(inputs, outputs, winner);
	} else if (taskCategory == 2) {
		std::array<double, 9> obs = carEnv.observe();
		std::vector<SnnIoEntry> inputs = {
			{"Posición X", TextFormat("%.2f m", obs[0])},
			{"Posición Y", TextFormat("%.2f m", obs[1])},
			{"Orientación", TextFormat("%.2f rad", obs[2])},
			{"Vel. X", TextFormat("%.2f m/s", obs[3])},
			{"Vel. Y", TextFormat("%.2f m/s", obs[4])},
			{"Vel. angular", TextFormat("%.2f rad/s", obs[5])},
			{"Lidar izquierdo", TextFormat("%.2f", obs[6])},
			{"Lidar centro", TextFormat("%.2f", obs[7])},
			{"Lidar derecho", TextFormat("%.2f", obs[8])},
		};
		std::vector<SnnIoEntry> outputs = {
			{"Acelerador/Freno", TextFormat("%.2f", pendingThrottle)},
			{"Dirección", TextFormat("%.2f", pendingSteering)},
		};
		net.setIoDisplay(inputs, outputs, -1);
	}
}

// Gamepad menu navigation: the d-pad moves a "focus" through whatever list of buttons the
// current screen passes in (nearest neighbor in the pressed direction), A "clicks" whichever
// button currently has focus, and B always triggers that screen's own "Volver" button,
// independent of focus. This reuses Button::isClicked()/draw()'s existing mouse-hover
// collision check unchanged: on a frame where the gamepad is driving, main() overrides the
// `mousePosition` fed to every isClicked()/draw() call for the rest of that frame to sit at
// the focused button's center, so the focused button lights up and can be "clicked" without
// any changes to Button itself. Moving the real mouse hands control back to it immediately.
struct GamepadMenuNav {
    int focusIndex = -1;
    int lastStatus = -1;
    bool gamepadActive = false;

    void beginFrame(int status, const std::vector<Rectangle>& rects, Vector2 mouseDelta) {
        if (status != lastStatus || focusIndex >= static_cast<int>(rects.size())) {
            lastStatus = status;
            focusIndex = rects.empty() ? -1 : 0;
        }
        if (mouseDelta.x != 0.0f || mouseDelta.y != 0.0f) gamepadActive = false;
        if (!IsGamepadAvailable(0) || rects.empty()) return;

        int dir = -1; // 0=up, 1=down, 2=left, 3=right
        if (IsGamepadButtonPressed(0, GAMEPAD_BUTTON_LEFT_FACE_UP)) dir = 0;
        else if (IsGamepadButtonPressed(0, GAMEPAD_BUTTON_LEFT_FACE_DOWN)) dir = 1;
        else if (IsGamepadButtonPressed(0, GAMEPAD_BUTTON_LEFT_FACE_LEFT)) dir = 2;
        else if (IsGamepadButtonPressed(0, GAMEPAD_BUTTON_LEFT_FACE_RIGHT)) dir = 3;
        if (dir < 0) return;

        gamepadActive = true;
        if (focusIndex < 0) { focusIndex = 0; return; }

        Vector2 from = { rects[static_cast<size_t>(focusIndex)].x + rects[static_cast<size_t>(focusIndex)].width / 2.0f,
                          rects[static_cast<size_t>(focusIndex)].y + rects[static_cast<size_t>(focusIndex)].height / 2.0f };
        int best = -1;
        float bestScore = 1e30f;
        for (size_t i = 0; i < rects.size(); ++i) {
            if (static_cast<int>(i) == focusIndex) continue;
            Vector2 c = { rects[i].x + rects[i].width / 2.0f, rects[i].y + rects[i].height / 2.0f };
            float dx = c.x - from.x, dy = c.y - from.y;
            bool matchesDir = (dir == 0 && dy < -1.0f) || (dir == 1 && dy > 1.0f) ||
                               (dir == 2 && dx < -1.0f) || (dir == 3 && dx > 1.0f);
            if (!matchesDir) continue;
            // Prefer candidates roughly aligned with the pressed axis over ones merely closer
            // in a diagonal sense (weighting the perpendicular offset higher than the primary
            // one keeps e.g. "down" from jumping sideways into a nearer-but-off-column button).
            float primary = (dir == 0 || dir == 1) ? std::fabs(dy) : std::fabs(dx);
            float perp = (dir == 0 || dir == 1) ? std::fabs(dx) : std::fabs(dy);
            float score = primary + perp * 3.0f;
            if (score < bestScore) { bestScore = score; best = static_cast<int>(i); }
        }
        if (best >= 0) focusIndex = best;
    }

    // Center of the focused rect once the gamepad is driving; the real mouse position
    // otherwise (no gamepad, nothing pressed yet this session, or the mouse just moved).
    Vector2 effectiveMouse(Vector2 realMouse, const std::vector<Rectangle>& rects) const {
        if (!gamepadActive || focusIndex < 0 || focusIndex >= static_cast<int>(rects.size())) return realMouse;
        const Rectangle& r = rects[static_cast<size_t>(focusIndex)];
        return { r.x + r.width / 2.0f, r.y + r.height / 2.0f };
    }

    bool confirmPressed() const {
        return gamepadActive && IsGamepadAvailable(0) && IsGamepadButtonPressed(0, GAMEPAD_BUTTON_RIGHT_FACE_DOWN);
    }
};

static bool gamepadBackPressed() {
    return IsGamepadAvailable(0) && IsGamepadButtonPressed(0, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT);
}

// True if `b` was clicked with the mouse, or if it's the button gamepad focus is currently
// sitting on (mousePosition already overridden to its center by GamepadMenuNav) and A/confirm
// was just pressed.
static bool activated(Button b, Vector2 mousePosition, bool gamepadConfirm) {
    return b.isClicked(mousePosition) || (gamepadConfirm && CheckCollisionPointRec(mousePosition, b.getButton()));
}

int main() {
	g_uiScale = SCREEN_WIDTH / 1680.0f;

	// Borderless window sized to exactly fill a monitor, rather than FLAG_FULLSCREEN_MODE:
	// that flag hands sizing to GLFW's "closest video mode on the primary monitor" logic,
	// which on a multi-monitor/scaled setup can pick the wrong monitor or a mismatched
	// resolution entirely (observed firsthand: it shrank the window instead of filling the
	// screen). Undecorated + positioned at the target monitor's origin has no such guesswork.
	SetConfigFlags(FLAG_WINDOW_UNDECORATED);
	InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "NeuroGame");
	int targetMonitor = 0;
	for (int m = 0; m < GetMonitorCount(); ++m) {
		if (GetMonitorWidth(m) == SCREEN_WIDTH && GetMonitorHeight(m) == SCREEN_HEIGHT) { targetMonitor = m; break; }
	}
	Vector2 targetMonitorPos = GetMonitorPosition(targetMonitor);
	SetWindowPosition(static_cast<int>(targetMonitorPos.x), static_cast<int>(targetMonitorPos.y));
	SetTargetFPS(60);

	// See gamepad_setup.cpp for why this specific controller needs a hand-derived mapping,
	// and why it must be called (once, right here) in every binary that reads gamepad input.
	SetupGameSirGamepadMapping();

	Menu menu = Menu(SCREEN_WIDTH, SCREEN_HEIGHT);
	bool quitRequested = false;

	// Every panel below used to reserve a flat "SCREEN_HEIGHT - 200" (120 top margin + 80
	// bottom margin); the 80px bottom margin no longer clears DrawSponsorLogos/
	// DrawFondecytCredit now that both got bigger, so it's now derived from
	// BrandingFooterHeight() (plus a little breathing room) instead of that old fixed 80.
	const float footerReserve = BrandingFooterHeight() + 20.0f;
	const float panelHeight = SCREEN_HEIGHT - 120.0f - footerReserve;

	// Network view layout: SNN on the left, task environment on the right.
	const Rectangle networkPanelBounds = { 80, 120, (SCREEN_WIDTH - 160 - 40) / 2.0f, panelHeight };
	const Rectangle envPanelBounds = {
		networkPanelBounds.x + networkPanelBounds.width + 40, 120,
		networkPanelBounds.width, panelHeight
	};
	// VS AI (Racing Car): no network shown, so the track gets the full width.
	const Rectangle racePanelBounds = { 80, 120, SCREEN_WIDTH - 160.0f, panelHeight };
	const Color HUMAN_CAR_COLOR = Color{220, 60, 60, 255};
	const Color AI_CAR_COLOR = Color{60, 110, 220, 255};
	// VS AI (Acrobot / Mountain Car): human's env on the left, AI's on the right, shorter
	// than the racing panel to leave room for the Izquierda/Derecha buttons at the bottom --
	// derived from those buttons' actual (already footer-aware) position rather than a fixed
	// guess, so the two can't drift out of sync again if either one's sizing changes.
	const float discreteBottomGap = 20.0f;
	const Rectangle discreteLeftPanelBounds = {
		80, 120, (SCREEN_WIDTH - 160 - 40) / 2.0f,
		menu.getLeftActionButton().getButton().y - discreteBottomGap - 120.0f
	};
	const Rectangle discreteRightPanelBounds = {
		discreteLeftPanelBounds.x + discreteLeftPanelBounds.width + 40, 120,
		discreteLeftPanelBounds.width, discreteLeftPanelBounds.height
	};
	// VS AI (Iris): network on the left, flower on the right, shorter still than the
	// Acrobot/Mountain Car panels to leave room for the guess buttons *and* the
	// reveal/prompt text line stacked below them (see IRIS_BUTTONS_ROW_Y/IRIS_TEXT_ROW_Y).
	const Rectangle irisLeftPanelBounds = { 80, 120, (SCREEN_WIDTH - 160 - 40) / 2.0f, panelHeight - 220.0f };
	const Rectangle irisRightPanelBounds = {
		irisLeftPanelBounds.x + irisLeftPanelBounds.width + 40, 120,
		irisLeftPanelBounds.width, irisLeftPanelBounds.height
	};
	// Guess/next-round buttons live in Menu (constructed at y=620 -- see menu.cpp); this
	// is just the text row drawn below them.
	constexpr float IRIS_TEXT_ROW_Y = 700.0f;

	SnnNetwork snnNetwork;
	int snnSpeedLevel = 0; // index into SNN_SPEED_FACTORS/SNN_SPEED_LABELS
	std::mt19937 rng(std::random_device{}());
	std::uniform_real_distribution<double> unit(0.0, 1.0);
	std::uniform_real_distribution<double> signedUnit(-1.0, 1.0);

	int selectedTaskCategory = -1;
	std::vector<SnnModelEntry> modelEntries;
	std::vector<Button> modelButtons;

	// Real closed loop: the network observes the actual environment state each window
	// and its decoded action drives the environment (task index: 0=Acrobot,
	// 1=Mountain Car, 2=Racing Car).
	int loadedTaskCategory = -1;
	AcrobotEnv acrobotEnv;
	MountainCarEnv mountainCarEnv;
	CarEnv carEnv;
	double pendingAction = 0.0;          // Acrobot / Mountain Car (single discrete action)
	double pendingThrottle = 0.0;        // Racing Car
	double pendingSteering = 0.0;        // Racing Car

	// VS AI: Racing Car -- human (red, keyboard) races an AI (blue, same closed loop as
	// above) on the same track. The network is never drawn here, but the AI still needs
	// its own SnnNetwork instance to decide -- kept separate from `snnNetwork` above so
	// the two modes never interfere with each other.
	SnnNetwork raceAiNetwork;
	CarEnv humanCarEnv;
	CarEnv aiCarEnv;
	double raceAiThrottle = 0.0;
	double raceAiSteering = 0.0;
	double racePhysicsAccumulatorMs = 0.0;

	// VS AI: Acrobot / Mountain Car -- same idea, but the human's action is discrete
	// (izquierda/derecha on-screen buttons -> -1/+1, same range the AI's decoder produces)
	// and each task keeps its own pair of env instances since they're unrelated worlds.
	AcrobotEnv humanAcrobotEnv;
	AcrobotEnv aiAcrobotEnv;
	MountainCarEnv humanMountainCarEnv;
	MountainCarEnv aiMountainCarEnv;
	double raceAiAction = 0.0;
	double raceDiscreteAccumulatorMs = 0.0;

	// VS AI: Iris -- a single shared flower (no dueling env instances): the human
	// clicks a species while the AI's network plays out its 20ms decision window, then
	// both guesses are revealed together against the flower's real species.
	enum class IrisPhase { Deciding, Revealing };
	constexpr double IRIS_ANIM_DURATION_MS = 900.0; // wall-clock time to play one 20ms window
	SnnNetwork irisNetwork;
	IrisEnv irisRound;
	IrisPhase irisPhase = IrisPhase::Deciding;
	int irisHumanGuess = -1;
	int irisAiGuess = -1;
	int irisHumanScore = 0;
	int irisAiScore = 0;
	int irisStreak = 0;
	double irisRoundStartTime = 0.0;    // GetTime() when the current flower was shown
	double irisHumanDecisionTimeSec = 0.0; // frozen the instant the human picks a species

	GamepadMenuNav gamepadNav;

	while (!WindowShouldClose() && !quitRequested) {
		// Mouse position
		Vector2 mousePosition = GetMousePosition();

		// Gamepad menu navigation: gather whichever buttons the current screen (and, for
		// LOAD_NETWORK_FILE_MENU / VS_AI_IRIS, its current sub-state) actually shows right
		// now, let the d-pad move focus among them, and -- if the gamepad is the one driving
		// this frame -- snap `mousePosition` to the focused button's center so every
		// isClicked()/draw() call below it (both the logic switch and the drawing section)
		// automatically treats it as hovered, with no per-button changes needed. The "Volver"
		// button on each screen is deliberately left out of this list: B reaches it directly
		// (see gamepadBackPressed()) instead of requiring it to be navigated to.
		std::vector<Rectangle> navRects;
		switch (menu.getStatus()) {
			case MAIN_MENU:
				navRects = { menu.getVsAI().getButton(), menu.getPlayNetwork().getButton(), menu.getQuit().getButton() };
				break;
			case VS_AI_MENU:
				navRects = { menu.getSelectAcrobot().getButton(), menu.getSelectMountainCar().getButton(),
					menu.getSelectRacingCar().getButton(), menu.getSelectIris().getButton() };
				break;
			case PLAY_NETWORK_MENU:
				navRects = { menu.getLoadNetwork().getButton(), menu.getCreateNetwork().getButton() };
				break;
			case LOAD_NETWORK_TASK_MENU:
				navRects = { menu.getLoadAcrobotTask().getButton(), menu.getLoadMountainCarTask().getButton(),
					menu.getLoadRacingCarTask().getButton() };
				break;
			case LOAD_NETWORK_FILE_MENU:
				for (Button& b : modelButtons) navRects.push_back(b.getButton());
				break;
			case VS_AI_IRIS:
				if (irisPhase == IrisPhase::Deciding && irisHumanGuess < 0) {
					navRects = { menu.getGuessSetosa().getButton(), menu.getGuessVersicolor().getButton(),
						menu.getGuessVirginica().getButton() };
				} else if (irisPhase == IrisPhase::Revealing) {
					navRects = { menu.getIrisNextRound().getButton() };
				}
				break;
			default:
				break;
		}
		gamepadNav.beginFrame(menu.getStatus(), navRects, GetMouseDelta());
		mousePosition = gamepadNav.effectiveMouse(mousePosition, navRects);
		bool gamepadConfirm = gamepadNav.confirmPressed();
		bool gamepadBack = gamepadBackPressed();

		switch (menu.getStatus()) {
			case MAIN_MENU:
				if (activated(menu.getVsAI(), mousePosition, gamepadConfirm)) {
					menu.setStatus(VS_AI_MENU);
				} else if (activated(menu.getPlayNetwork(), mousePosition, gamepadConfirm)) {
					menu.setStatus(PLAY_NETWORK_MENU);
				} else if (activated(menu.getQuit(), mousePosition, gamepadConfirm)) {
					quitRequested = true;
				}
				break;

			case VS_AI_MENU:
				if (activated(menu.getSelectAcrobot(), mousePosition, gamepadConfirm)) {
					std::vector<SnnModelEntry> models = listSnnModels("models/acrobot");
					if (!models.empty() &&
						raceAiNetwork.load(models[0].outPath, models[0].wiPath, snnAcrobotPreset(), {0, 0, 100, 100})) {
						humanAcrobotEnv.reset(rng);
						aiAcrobotEnv.reset(rng);
						std::array<double, 6> obs = aiAcrobotEnv.observe();
						raceAiNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
						raceAiAction = raceAiNetwork.decodeFirstSpikeWinner() - 1.0;
						raceDiscreteAccumulatorMs = 0.0;
						menu.setStatus(VS_AI_ACROBOT);
					}
				} else if (activated(menu.getSelectMountainCar(), mousePosition, gamepadConfirm)) {
					std::vector<SnnModelEntry> models = listSnnModels("models/mountain_car");
					if (!models.empty() &&
						raceAiNetwork.load(models[0].outPath, models[0].wiPath, snnMountainCarPreset(), {0, 0, 100, 100})) {
						humanMountainCarEnv.reset(rng);
						aiMountainCarEnv.reset(rng);
						std::array<double, 2> obs = aiMountainCarEnv.observe();
						raceAiNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
						raceAiAction = raceAiNetwork.decodeFirstSpikeWinner() - 1.0;
						raceDiscreteAccumulatorMs = 0.0;
						menu.setStatus(VS_AI_MOUNTAIN_CAR);
					}
				} else if (activated(menu.getSelectRacingCar(), mousePosition, gamepadConfirm)) {
					std::vector<SnnModelEntry> raceModels = listSnnModels("models/racing_car");
					if (!raceModels.empty() &&
						raceAiNetwork.load(raceModels[0].outPath, raceModels[0].wiPath, snnRacingCarPreset(), {0, 0, 100, 100})) {
						humanCarEnv.reset(rng);
						aiCarEnv.reset(rng);
						raceAiNetwork.simulateStep(encodeCarObservation(aiCarEnv.observe()));
						raceAiThrottle = decodeCarContinuousAction(raceAiNetwork, 0);
						raceAiSteering = decodeCarContinuousAction(raceAiNetwork, 1);
						racePhysicsAccumulatorMs = 0.0;
						menu.setStatus(VS_AI_RACING_CAR);
					}
				} else if (activated(menu.getSelectIris(), mousePosition, gamepadConfirm)) {
					std::vector<SnnModelEntry> models = listSnnModels("models/iris");
					if (!models.empty() &&
						irisNetwork.load(models[0].outPath, models[0].wiPath, snnIrisPreset(), irisLeftPanelBounds)) {
						irisRound.pickRandom(rng);
						std::array<double, 4> obs = irisRound.observe();
						irisNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
						irisPhase = IrisPhase::Deciding;
						irisHumanGuess = -1;
						irisAiGuess = -1;
						irisRoundStartTime = GetTime();
						menu.setStatus(VS_AI_IRIS);
					}
				} else if (menu.getBackFromVsAI().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(MAIN_MENU);
				}
				break;

			case PLAY_NETWORK_MENU:
				if (activated(menu.getLoadNetwork(), mousePosition, gamepadConfirm)) {
					menu.setStatus(LOAD_NETWORK_TASK_MENU);
				} else if (activated(menu.getCreateNetwork(), mousePosition, gamepadConfirm)) {
					// TODO: iniciar flujo de creación de una nueva red
				} else if (menu.getBackFromPlayNetwork().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(MAIN_MENU);
				}
				break;

			case LOAD_NETWORK_TASK_MENU: {
				int clickedTask = -1;
				if (activated(menu.getLoadAcrobotTask(), mousePosition, gamepadConfirm)) clickedTask = 0;
				else if (activated(menu.getLoadMountainCarTask(), mousePosition, gamepadConfirm)) clickedTask = 1;
				else if (activated(menu.getLoadRacingCarTask(), mousePosition, gamepadConfirm)) clickedTask = 2;

				if (clickedTask >= 0) {
					selectedTaskCategory = clickedTask;
					modelEntries = listSnnModels(snnTaskCategories()[clickedTask].directory);
					modelButtons = buildModelButtons(modelEntries);
					menu.setStatus(LOAD_NETWORK_FILE_MENU);
				} else if (menu.getBackFromLoadTask().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(PLAY_NETWORK_MENU);
				}
				break;
			}

			case LOAD_NETWORK_FILE_MENU:
				if (menu.getBackFromLoadFile().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(LOAD_NETWORK_TASK_MENU);
					break;
				}
				for (size_t i = 0; i < modelButtons.size(); ++i) {
					if (!activated(modelButtons[i], mousePosition, gamepadConfirm)) continue;
					const SnnTaskCategory& category = snnTaskCategories()[static_cast<size_t>(selectedTaskCategory)];
					if (snnNetwork.load(modelEntries[i].outPath, modelEntries[i].wiPath, category.preset(), networkPanelBounds)) {
						loadedTaskCategory = selectedTaskCategory;
						if (loadedTaskCategory == 0) {
							acrobotEnv.reset(rng);
							std::array<double, 6> obs = acrobotEnv.observe();
							snnNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
							pendingAction = snnNetwork.decodeFirstSpikeWinner() - 1.0;
						} else if (loadedTaskCategory == 1) {
							mountainCarEnv.reset(rng);
							std::array<double, 2> obs = mountainCarEnv.observe();
							snnNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
							pendingAction = snnNetwork.decodeFirstSpikeWinner() - 1.0;
						} else if (loadedTaskCategory == 2) {
							carEnv.reset(rng);
							snnNetwork.simulateStep(encodeCarObservation(carEnv.observe()));
							pendingThrottle = decodeCarContinuousAction(snnNetwork, 0);
							pendingSteering = decodeCarContinuousAction(snnNetwork, 1);
						}
						menu.setStatus(NETWORK_VIEW);
					}
					break;
				}
				break;

			case NETWORK_VIEW: {
				if (menu.getBackFromNetworkView().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(PLAY_NETWORK_MENU);
					break;
				}
				if (IsKeyPressed(KEY_SPACE)) {
					snnSpeedLevel = (snnSpeedLevel + 1) % SNN_SPEED_LEVEL_COUNT;
				}

				// Real closed loop: apply the action decoded from the window that just
				// finished playing, observe the resulting state, and feed it back in for
				// the next window (SMALL: raw values; TTFS: normalized to [0,1]).
				auto takeEnvStep = [&]() {
					if (loadedTaskCategory == 0) {
						acrobotEnv.step(pendingAction, rng);
						std::array<double, 6> obs = acrobotEnv.observe();
						snnNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
						pendingAction = snnNetwork.decodeFirstSpikeWinner() - 1.0;
					} else if (loadedTaskCategory == 1) {
						mountainCarEnv.step(pendingAction, rng);
						std::array<double, 2> obs = mountainCarEnv.observe();
						snnNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
						pendingAction = snnNetwork.decodeFirstSpikeWinner() - 1.0;
					} else if (loadedTaskCategory == 2) {
						carEnv.step(pendingThrottle, pendingSteering, rng);
						snnNetwork.simulateStep(encodeCarObservation(carEnv.observe()));
						pendingThrottle = decodeCarContinuousAction(snnNetwork, 0);
						pendingSteering = decodeCarContinuousAction(snnNetwork, 1);
					} else {
						std::vector<double> observation(static_cast<size_t>(snnNetwork.syntheticInputSize()));
						bool isSmallEncoder = snnNetwork.encoderKind() == SnnEncoderKind::Small;
						for (double& value : observation) value = isSmallEncoder ? signedUnit(rng) : unit(rng);
						snnNetwork.simulateStep(observation);
					}
				};

				double realStepMs = realStepMsForTask(loadedTaskCategory) * SNN_SPEED_FACTORS[snnSpeedLevel];
				double speed = SnnNetwork::SIM_WINDOW_MS / realStepMs; // sim-ms per wall-ms

				// A single advance() per rendered frame (60 FPS) can't keep up with tasks
				// whose real step rate exceeds the render rate -- Racing Car needs ~100
				// decisions/s, same as VS AI mode's physics accumulator needs, so use the
				// same pattern here: catch up with as many steps as elapsed wall-clock time
				// actually calls for, capped so a stall can't spiral.
				double remainingWallMs = std::min(GetFrameTime() * 1000.0, realStepMs * 10.0);
				while (remainingWallMs > 1e-9) {
					if (snnNetwork.isStepFinished()) takeEnvStep();
					double simMsLeftInWindow = (1.0 - snnNetwork.progress()) * SnnNetwork::SIM_WINDOW_MS;
					double wallMsThisIter = std::min(remainingWallMs, simMsLeftInWindow / speed);
					snnNetwork.advance(wallMsThisIter * speed);
					remainingWallMs -= wallMsThisIter;
				}
				break;
			}

			case VS_AI_RACING_CAR: {
				if (menu.getBackFromVsAiRacingCar().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(VS_AI_MENU);
					break;
				}

				double humanThrottle = 0.0;
				double humanSteering = 0.0;

				// Gamepad (analog): left stick steers, triggers accelerate/brake. Raylib
				// reports GAMEPAD_AXIS_LEFT_X as -1 (left) .. +1 (right) and the trigger
				// axes as -1 (released) .. +1 (fully pressed); if the trigger direction
				// feels backwards on a given controller/driver, flip the (+1)/2 sign below.
				if (IsGamepadAvailable(0)) {
					constexpr float STICK_DEADZONE = 0.12f;
					float stickX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X);
					if (stickX < -STICK_DEADZONE || stickX > STICK_DEADZONE) {
						humanSteering = -static_cast<double>(stickX); // stick left(-) -> steer +1 (turn left)
					}

					float rightTrigger = GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_TRIGGER); // accelerate
					float leftTrigger = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_TRIGGER);    // brake/reverse
					double accel = std::clamp((static_cast<double>(rightTrigger) + 1.0) / 2.0, 0.0, 1.0);
					double brake = std::clamp((static_cast<double>(leftTrigger) + 1.0) / 2.0, 0.0, 1.0);
					humanThrottle = accel - brake;
				}

				// Keyboard (digital): overrides the gamepad whenever a relevant key is
				// actually held, so both input methods keep working interchangeably.
				if (IsKeyDown(KEY_UP)) humanThrottle = 1.0;
				else if (IsKeyDown(KEY_DOWN)) humanThrottle = -1.0;
				// Positive steering increases mu (a CCW/left turn on screen) -- see
				// car_env.cpp's bicycle model, verified empirically.
				if (IsKeyDown(KEY_LEFT)) humanSteering = 1.0;
				else if (IsKeyDown(KEY_RIGHT)) humanSteering = -1.0;

				humanThrottle = std::clamp(humanThrottle, -1.0, 1.0);
				humanSteering = std::clamp(humanSteering, -1.0, 1.0);

				// Both cars run on the same fixed-rate physics clock (CarEnv::DT = 0.01s,
				// matching the AI's real decision cadence -- one AI decision per physics
				// tick, exactly like SnnCarTask.cpp's training/eval loop). The network
				// isn't shown, so there's no reason to spread it over wall-clock time the
				// way NETWORK_VIEW does -- just run it to completion every tick.
				const double physicsDtMs = CarEnv::DT * 1000.0;
				racePhysicsAccumulatorMs += GetFrameTime() * 1000.0;
				racePhysicsAccumulatorMs = std::min(racePhysicsAccumulatorMs, physicsDtMs * 10.0);
				while (racePhysicsAccumulatorMs >= physicsDtMs) {
					humanCarEnv.step(humanThrottle, humanSteering, rng);

					aiCarEnv.step(raceAiThrottle, raceAiSteering, rng);
					raceAiNetwork.simulateStep(encodeCarObservation(aiCarEnv.observe()));
					raceAiThrottle = decodeCarContinuousAction(raceAiNetwork, 0);
					raceAiSteering = decodeCarContinuousAction(raceAiNetwork, 1);

					racePhysicsAccumulatorMs -= physicsDtMs;
				}
				break;
			}

			case VS_AI_ACROBOT: {
				if (menu.getBackFromVsAiAcrobot().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(VS_AI_MENU);
					break;
				}

				// Gamepad (analog): left stick X thresholded into a discrete left/right
				// action -- reuses the same leftx mapping verified for VS_AI_RACING_CAR since
				// this controller's SetGamepadMappings entry above has no buttons mapped.
				double humanAction = 0.0;
				if (IsGamepadAvailable(0)) {
					constexpr float STICK_DEADZONE = 0.12f;
					float stickX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X);
					if (stickX < -STICK_DEADZONE) humanAction = -1.0;
					else if (stickX > STICK_DEADZONE) humanAction = 1.0;
				}
				// Mouse (digital): overrides the gamepad whenever a button is actually held.
				if (menu.getLeftActionButton().isBeingClicked(mousePosition)) humanAction = -1.0;
				else if (menu.getRightActionButton().isBeingClicked(mousePosition)) humanAction = 1.0;

				const double physicsDtMs = realStepMsForTask(0); // Acrobot: DT = 0.2s
				raceDiscreteAccumulatorMs += GetFrameTime() * 1000.0;
				raceDiscreteAccumulatorMs = std::min(raceDiscreteAccumulatorMs, physicsDtMs * 10.0);
				while (raceDiscreteAccumulatorMs >= physicsDtMs) {
					humanAcrobotEnv.step(humanAction, rng);

					aiAcrobotEnv.step(raceAiAction, rng);
					std::array<double, 6> obs = aiAcrobotEnv.observe();
					raceAiNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
					raceAiAction = raceAiNetwork.decodeFirstSpikeWinner() - 1.0;

					raceDiscreteAccumulatorMs -= physicsDtMs;
				}
				break;
			}

			case VS_AI_MOUNTAIN_CAR: {
				if (menu.getBackFromVsAiMountainCar().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(VS_AI_MENU);
					break;
				}

				// Gamepad (analog): left stick X thresholded into a discrete left/right
				// action -- reuses the same leftx mapping verified for VS_AI_RACING_CAR since
				// this controller's SetGamepadMappings entry above has no buttons mapped.
				double humanAction = 0.0;
				if (IsGamepadAvailable(0)) {
					constexpr float STICK_DEADZONE = 0.12f;
					float stickX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X);
					if (stickX < -STICK_DEADZONE) humanAction = -1.0;
					else if (stickX > STICK_DEADZONE) humanAction = 1.0;
				}
				// Mouse (digital): overrides the gamepad whenever a button is actually held.
				if (menu.getLeftActionButton().isBeingClicked(mousePosition)) humanAction = -1.0;
				else if (menu.getRightActionButton().isBeingClicked(mousePosition)) humanAction = 1.0;

				const double physicsDtMs = realStepMsForTask(1); // Mountain Car: ~30 fps convention
				raceDiscreteAccumulatorMs += GetFrameTime() * 1000.0;
				raceDiscreteAccumulatorMs = std::min(raceDiscreteAccumulatorMs, physicsDtMs * 10.0);
				while (raceDiscreteAccumulatorMs >= physicsDtMs) {
					humanMountainCarEnv.step(humanAction, rng);

					aiMountainCarEnv.step(raceAiAction, rng);
					std::array<double, 2> obs = aiMountainCarEnv.observe();
					raceAiNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
					raceAiAction = raceAiNetwork.decodeFirstSpikeWinner() - 1.0;

					raceDiscreteAccumulatorMs -= physicsDtMs;
				}
				break;
			}

			case VS_AI_IRIS: {
				if (menu.getBackFromVsAiIris().isClicked(mousePosition) || gamepadBack) {
					menu.setStatus(VS_AI_MENU);
					break;
				}

				if (irisPhase == IrisPhase::Deciding) {
					if (!irisNetwork.isStepFinished()) {
						double speed = SnnNetwork::SIM_WINDOW_MS / IRIS_ANIM_DURATION_MS; // sim-ms per wall-ms
						irisNetwork.advance(GetFrameTime() * 1000.0 * speed);
					}

					// A guess is final the instant it's clicked -- this is also the moment
					// the decision-time stopwatch freezes, so re-clicking a different
					// species afterward (while still waiting on the AI's animation) can't
					// change it.
					if (irisHumanGuess < 0) {
						if (activated(menu.getGuessSetosa(), mousePosition, gamepadConfirm)) irisHumanGuess = 0;
						else if (activated(menu.getGuessVersicolor(), mousePosition, gamepadConfirm)) irisHumanGuess = 1;
						else if (activated(menu.getGuessVirginica(), mousePosition, gamepadConfirm)) irisHumanGuess = 2;
						if (irisHumanGuess >= 0) irisHumanDecisionTimeSec = GetTime() - irisRoundStartTime;
					}

					if (irisHumanGuess >= 0 && irisNetwork.isStepFinished()) {
						irisAiGuess = irisNetwork.decodeFirstSpikeWinner();
						bool humanCorrect = irisHumanGuess == irisRound.trueLabel();
						bool aiCorrect = irisAiGuess == irisRound.trueLabel();
						if (humanCorrect) { ++irisHumanScore; ++irisStreak; } else { irisStreak = 0; }
						if (aiCorrect) ++irisAiScore;
						irisPhase = IrisPhase::Revealing;
					}
				} else { // Revealing
					if (activated(menu.getIrisNextRound(), mousePosition, gamepadConfirm)) {
						irisRound.pickRandom(rng);
						std::array<double, 4> obs = irisRound.observe();
						irisNetwork.simulateStep(std::vector<double>(obs.begin(), obs.end()));
						irisHumanGuess = -1;
						irisAiGuess = -1;
						irisRoundStartTime = GetTime();
						irisPhase = IrisPhase::Deciding;
					}
				}
				break;
			}
		}

		// Drawing
		BeginDrawing();
		ClearBackground(RAYWHITE);
		if (menu.getStatus() == NETWORK_VIEW) {
			if (loadedTaskCategory >= 0 && loadedTaskCategory <= 2) {
				updateSnnIoDisplay(snnNetwork, loadedTaskCategory, acrobotEnv, mountainCarEnv, carEnv,
					pendingAction, pendingThrottle, pendingSteering);
			}
			snnNetwork.draw(networkPanelBounds);
			DrawText("Red neuronal", static_cast<int>(networkPanelBounds.x), 95, FS(20), DARKGRAY);

			if (loadedTaskCategory == 0) {
				acrobotEnv.draw(envPanelBounds, snnNetwork.progress());
				DrawText("Entorno: Acrobot", static_cast<int>(envPanelBounds.x), 95, FS(20), DARKGRAY);
			} else if (loadedTaskCategory == 1) {
				mountainCarEnv.draw(envPanelBounds);
				DrawText("Entorno: Mountain Car", static_cast<int>(envPanelBounds.x), 95, FS(20), DARKGRAY);
			} else if (loadedTaskCategory == 2) {
				carEnv.draw(envPanelBounds);
				DrawText("Entorno: Racing Car", static_cast<int>(envPanelBounds.x), 95, FS(20), DARKGRAY);
				DrawText(TextFormat("Pasos: %d", carEnv.stepCount()),
					static_cast<int>(envPanelBounds.x), static_cast<int>(envPanelBounds.y + envPanelBounds.height - 20), FS(18), DARKGRAY);
			} else {
				const char* placeholder = "Entorno aún no implementado para esta tarea";
				int placeholderWidth = MeasureText(placeholder, FS(20));
				DrawText(placeholder,
					static_cast<int>(envPanelBounds.x + envPanelBounds.width / 2.0f - placeholderWidth / 2.0f),
					static_cast<int>(envPanelBounds.y + envPanelBounds.height / 2.0f), FS(20), GRAY);
			}

			float dividerX = networkPanelBounds.x + networkPanelBounds.width + 20;
			DrawLineEx({ dividerX, networkPanelBounds.y }, { dividerX, networkPanelBounds.y + networkPanelBounds.height }, 1.0f, LIGHTGRAY);

			int nextSpeedLevel = (snnSpeedLevel + 1) % SNN_SPEED_LEVEL_COUNT;
			DrawText(TextFormat("Velocidad: %s (ESPACIO para %s)",
					SNN_SPEED_LABELS[snnSpeedLevel], SNN_SPEED_LABELS[nextSpeedLevel]),
				20, 70, FS(20), DARKGRAY);
		} else if (menu.getStatus() == VS_AI_RACING_CAR) {
			aiCarEnv.drawTrack(racePanelBounds);
			humanCarEnv.drawCar(racePanelBounds, HUMAN_CAR_COLOR, false);
			aiCarEnv.drawCar(racePanelBounds, AI_CAR_COLOR, true);

			DrawText("Tú (rojo) vs IA (azul)", static_cast<int>(racePanelBounds.x), 90, FS(22), DARKGRAY);
			DrawText("Flechas o gamepad: arriba/abajo o gatillos acelerar/frenar, izquierda/derecha o stick izquierdo girar",
				static_cast<int>(racePanelBounds.x),
				static_cast<int>(racePanelBounds.y + racePanelBounds.height + 8), FS(18), GRAY);
		} else if (menu.getStatus() == VS_AI_ACROBOT) {
			// Same start/mid/end sub-step interpolation as "Evaluar red": animate each
			// pendulum smoothly across the just-taken step's motion for the whole interval
			// leading up to the next one, instead of snapping straight to the new pose.
			double acrobotProgress = raceDiscreteAccumulatorMs / realStepMsForTask(0);
			humanAcrobotEnv.draw(discreteLeftPanelBounds, acrobotProgress);
			aiAcrobotEnv.draw(discreteRightPanelBounds, acrobotProgress);
			DrawText("Tú", static_cast<int>(discreteLeftPanelBounds.x), 95, FS(20), DARKGRAY);
			DrawText("IA", static_cast<int>(discreteRightPanelBounds.x), 95, FS(20), DARKGRAY);
			float acrobotDividerX = discreteLeftPanelBounds.x + discreteLeftPanelBounds.width + 20;
			DrawLineEx({ acrobotDividerX, discreteLeftPanelBounds.y }, { acrobotDividerX, discreteLeftPanelBounds.y + discreteLeftPanelBounds.height }, 1.0f, LIGHTGRAY);
			DrawText("Mantén presionado Izquierda o Derecha para aplicar torque",
				static_cast<int>(discreteLeftPanelBounds.x), static_cast<int>(discreteLeftPanelBounds.y + discreteLeftPanelBounds.height + 12), FS(18), GRAY);
		} else if (menu.getStatus() == VS_AI_MOUNTAIN_CAR) {
			humanMountainCarEnv.draw(discreteLeftPanelBounds);
			aiMountainCarEnv.draw(discreteRightPanelBounds);
			DrawText("Tú", static_cast<int>(discreteLeftPanelBounds.x), 95, FS(20), DARKGRAY);
			DrawText("IA", static_cast<int>(discreteRightPanelBounds.x), 95, FS(20), DARKGRAY);
			float mcDividerX = discreteLeftPanelBounds.x + discreteLeftPanelBounds.width + 20;
			DrawLineEx({ mcDividerX, discreteLeftPanelBounds.y }, { mcDividerX, discreteLeftPanelBounds.y + discreteLeftPanelBounds.height }, 1.0f, LIGHTGRAY);
			DrawText("Mantén presionado Izquierda o Derecha para empujar el auto",
				static_cast<int>(discreteLeftPanelBounds.x), static_cast<int>(discreteLeftPanelBounds.y + discreteLeftPanelBounds.height + 12), FS(18), GRAY);
		} else if (menu.getStatus() == VS_AI_IRIS) {
			// The AI's decoded species stays hidden (no highlighted output/legend entry)
			// until the round is revealed, even if its spike window finishes first --
			// otherwise a fast-finishing network would spoil the answer before the human
			// has picked.
			int winnerHighlight = (irisPhase == IrisPhase::Revealing) ? irisAiGuess : -1;
			const IrisSample& irisSample = irisRound.sample();
			std::vector<SnnIoEntry> inputs = {
				{"Sepalo largo", TextFormat("%.1f cm", irisSample.sepalLength)},
				{"Sepalo ancho", TextFormat("%.1f cm", irisSample.sepalWidth)},
				{"Petalo largo", TextFormat("%.1f cm", irisSample.petalLength)},
				{"Petalo ancho", TextFormat("%.1f cm", irisSample.petalWidth)},
			};
			std::vector<SnnIoEntry> outputs = {
				{"Setosa", (winnerHighlight == 0) ? "ACTIVA" : ""},
				{"Versicolor", (winnerHighlight == 1) ? "ACTIVA" : ""},
				{"Virginica", (winnerHighlight == 2) ? "ACTIVA" : ""},
			};
			irisNetwork.setIoDisplay(inputs, outputs, winnerHighlight);
			irisNetwork.draw(irisLeftPanelBounds);
			DrawText("Red neuronal (IA)", static_cast<int>(irisLeftPanelBounds.x), 95, FS(20), DARKGRAY);

			irisRound.draw(irisRightPanelBounds);
			DrawText("Adivina la flor", static_cast<int>(irisRightPanelBounds.x), 95, FS(20), DARKGRAY);

			float dividerX = irisLeftPanelBounds.x + irisLeftPanelBounds.width + 20;
			DrawLineEx({ dividerX, irisLeftPanelBounds.y }, { dividerX, irisLeftPanelBounds.y + irisLeftPanelBounds.height }, 1.0f, LIGHTGRAY);

			std::string scoreText = TextFormat("Tú: %d   IA: %d   Racha: %d", irisHumanScore, irisAiScore, irisStreak);
			int scoreWidth = MeasureText(scoreText.c_str(), FS(22));
			DrawText(scoreText.c_str(), SCREEN_WIDTH / 2 - scoreWidth / 2, 60, FS(22), DARKGRAY);

			if (irisPhase == IrisPhase::Deciding) {
				// Drawn manually (not by Menu::draw()) so they can be hidden once revealed
				// instead of sitting inert on top of the reveal text/next-round button.
				menu.getGuessSetosa().draw(mousePosition);
				menu.getGuessVersicolor().draw(mousePosition);
				menu.getGuessVirginica().draw(mousePosition);

				// Live stopwatch: keeps counting up until the guess is locked in, then
				// holds at irisHumanDecisionTimeSec while the AI's animation finishes.
				double elapsedSec = (irisHumanGuess < 0) ? (GetTime() - irisRoundStartTime) : irisHumanDecisionTimeSec;
				std::string prompt = (irisHumanGuess < 0)
					? TextFormat("Elegí una especie mientras la red decide... (%.1fs)", elapsedSec)
					: TextFormat("Elegiste en %.1fs. Esperando a la IA...", elapsedSec);
				int promptWidth = MeasureText(prompt.c_str(), FS(18));
				DrawText(prompt.c_str(), SCREEN_WIDTH / 2 - promptWidth / 2, static_cast<int>(IRIS_TEXT_ROW_Y), FS(18), GRAY);
			} else {
				bool humanCorrect = irisHumanGuess == irisRound.trueLabel();
				bool aiCorrect = irisAiGuess == irisRound.trueLabel();
				std::string reveal = TextFormat("Real: %s   |   Vos: %s %s en %.1fs   |   IA: %s %s",
					IRIS_SPECIES_NAMES[irisRound.trueLabel()],
					IRIS_SPECIES_NAMES[irisHumanGuess], humanCorrect ? "(correcto)" : "(incorrecto)", irisHumanDecisionTimeSec,
					IRIS_SPECIES_NAMES[irisAiGuess], aiCorrect ? "(correcto)" : "(incorrecto)");
				int revealWidth = MeasureText(reveal.c_str(), FS(20));
				DrawText(reveal.c_str(), SCREEN_WIDTH / 2 - revealWidth / 2, static_cast<int>(IRIS_TEXT_ROW_Y), FS(20),
					(humanCorrect ? Color{40, 150, 70, 255} : Color{200, 60, 60, 255}));
				menu.getIrisNextRound().draw(mousePosition);
			}
		} else if (menu.getStatus() == LOAD_NETWORK_FILE_MENU) {
			const std::string& taskLabel = snnTaskCategories()[static_cast<size_t>(selectedTaskCategory)].label;
			std::string title = "Modelos disponibles: " + taskLabel;
			int titleWidth = MeasureText(title.c_str(), FS(24));
			DrawText(title.c_str(), SCREEN_WIDTH / 2 - titleWidth / 2, 80, FS(24), DARKGRAY);

			if (modelButtons.empty()) {
				const char* empty = "No hay modelos guardados para esta tarea todavía.";
				int emptyWidth = MeasureText(empty, FS(20));
				DrawText(empty, SCREEN_WIDTH / 2 - emptyWidth / 2, SCREEN_HEIGHT / 2, FS(20), GRAY);
			}
			for (Button& button : modelButtons) button.draw(mousePosition);
		}
		menu.draw(mousePosition);
		if (menu.getStatus() == VS_AI_ACROBOT || menu.getStatus() == VS_AI_MOUNTAIN_CAR) {
			drawArrowIcon(menu.getLeftActionButton().getButton(), false, DARKGRAY);
			drawArrowIcon(menu.getRightActionButton().getButton(), true, DARKGRAY);
		}
		DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
		DrawFondecytCredit(SCREEN_WIDTH, SCREEN_HEIGHT);
		EndDrawing();
	}

	CloseWindow();

	return 0;
}
