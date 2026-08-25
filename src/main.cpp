#include <raylib.h>
#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <vector>

#include "../include/acrobot_env.hpp"
#include "../include/branding.hpp"
#include "../include/car_env.hpp"
#include "../include/menu.hpp"
#include "../include/model_browser.hpp"
#include "../include/mountain_car_env.hpp"
#include "../include/snn_network.hpp"

// Screen dimensions
#define SCREEN_WIDTH 1680
#define SCREEN_HEIGHT 900

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
			{"Posicion", TextFormat("%.2f", obs[0])},
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
			{"Posicion X", TextFormat("%.2f m", obs[0])},
			{"Posicion Y", TextFormat("%.2f m", obs[1])},
			{"Orientacion", TextFormat("%.2f rad", obs[2])},
			{"Vel. X", TextFormat("%.2f m/s", obs[3])},
			{"Vel. Y", TextFormat("%.2f m/s", obs[4])},
			{"Vel. angular", TextFormat("%.2f rad/s", obs[5])},
			{"Lidar izquierdo", TextFormat("%.2f", obs[6])},
			{"Lidar centro", TextFormat("%.2f", obs[7])},
			{"Lidar derecho", TextFormat("%.2f", obs[8])},
		};
		std::vector<SnnIoEntry> outputs = {
			{"Acelerador/Freno", TextFormat("%.2f", pendingThrottle)},
			{"Direccion", TextFormat("%.2f", pendingSteering)},
		};
		net.setIoDisplay(inputs, outputs, -1);
	}
}

int main() {
	InitWindow(SCREEN_WIDTH,SCREEN_HEIGHT, "NeuroGame");
	SetTargetFPS(60);

	Menu menu = Menu(SCREEN_WIDTH, SCREEN_HEIGHT);
	bool quitRequested = false;

	// Network view layout: SNN on the left, task environment on the right.
	const Rectangle networkPanelBounds = { 80, 120, (SCREEN_WIDTH - 160 - 40) / 2.0f, SCREEN_HEIGHT - 200 };
	const Rectangle envPanelBounds = {
		networkPanelBounds.x + networkPanelBounds.width + 40, 120,
		networkPanelBounds.width, SCREEN_HEIGHT - 200
	};
	// VS AI (Racing Car): no network shown, so the track gets the full width.
	const Rectangle racePanelBounds = { 80, 120, SCREEN_WIDTH - 160.0f, SCREEN_HEIGHT - 200.0f };
	const Color HUMAN_CAR_COLOR = Color{220, 60, 60, 255};
	const Color AI_CAR_COLOR = Color{60, 110, 220, 255};
	// VS AI (Acrobot / Mountain Car): human's env on the left, AI's on the right, shorter
	// than the racing panel to leave room for the Izquierda/Derecha buttons at the bottom.
	const Rectangle discreteLeftPanelBounds = { 80, 120, (SCREEN_WIDTH - 160 - 40) / 2.0f, SCREEN_HEIGHT - 200 - 140.0f };
	const Rectangle discreteRightPanelBounds = {
		discreteLeftPanelBounds.x + discreteLeftPanelBounds.width + 40, 120,
		discreteLeftPanelBounds.width, discreteLeftPanelBounds.height
	};

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

	while (!WindowShouldClose() && !quitRequested) {
		// Mouse position
		Vector2 mousePosition = GetMousePosition();

		switch (menu.getStatus()) {
			case MAIN_MENU:
				if (menu.getVsAI().isClicked(mousePosition)) {
					menu.setStatus(VS_AI_MENU);
				} else if (menu.getPlayNetwork().isClicked(mousePosition)) {
					menu.setStatus(PLAY_NETWORK_MENU);
				} else if (menu.getQuit().isClicked(mousePosition)) {
					quitRequested = true;
				}
				break;

			case VS_AI_MENU:
				if (menu.getSelectAcrobot().isClicked(mousePosition)) {
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
				} else if (menu.getSelectMountainCar().isClicked(mousePosition)) {
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
				} else if (menu.getSelectRacingCar().isClicked(mousePosition)) {
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
				} else if (menu.getBackFromVsAI().isClicked(mousePosition)) {
					menu.setStatus(MAIN_MENU);
				}
				break;

			case PLAY_NETWORK_MENU:
				if (menu.getLoadNetwork().isClicked(mousePosition)) {
					menu.setStatus(LOAD_NETWORK_TASK_MENU);
				} else if (menu.getCreateNetwork().isClicked(mousePosition)) {
					// TODO: iniciar flujo de creación de una nueva red
				} else if (menu.getBackFromPlayNetwork().isClicked(mousePosition)) {
					menu.setStatus(MAIN_MENU);
				}
				break;

			case LOAD_NETWORK_TASK_MENU: {
				int clickedTask = -1;
				if (menu.getLoadAcrobotTask().isClicked(mousePosition)) clickedTask = 0;
				else if (menu.getLoadMountainCarTask().isClicked(mousePosition)) clickedTask = 1;
				else if (menu.getLoadRacingCarTask().isClicked(mousePosition)) clickedTask = 2;

				if (clickedTask >= 0) {
					selectedTaskCategory = clickedTask;
					modelEntries = listSnnModels(snnTaskCategories()[clickedTask].directory);
					modelButtons = buildModelButtons(modelEntries);
					menu.setStatus(LOAD_NETWORK_FILE_MENU);
				} else if (menu.getBackFromLoadTask().isClicked(mousePosition)) {
					menu.setStatus(PLAY_NETWORK_MENU);
				}
				break;
			}

			case LOAD_NETWORK_FILE_MENU:
				if (menu.getBackFromLoadFile().isClicked(mousePosition)) {
					menu.setStatus(LOAD_NETWORK_TASK_MENU);
					break;
				}
				for (size_t i = 0; i < modelButtons.size(); ++i) {
					if (!modelButtons[i].isClicked(mousePosition)) continue;
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
				if (menu.getBackFromNetworkView().isClicked(mousePosition)) {
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
				if (menu.getBackFromVsAiRacingCar().isClicked(mousePosition)) {
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
				if (menu.getBackFromVsAiAcrobot().isClicked(mousePosition)) {
					menu.setStatus(VS_AI_MENU);
					break;
				}

				double humanAction = 0.0;
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
				if (menu.getBackFromVsAiMountainCar().isClicked(mousePosition)) {
					menu.setStatus(VS_AI_MENU);
					break;
				}

				double humanAction = 0.0;
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
			DrawText("Red neuronal", static_cast<int>(networkPanelBounds.x), 95, 20, DARKGRAY);

			if (loadedTaskCategory == 0) {
				acrobotEnv.draw(envPanelBounds, snnNetwork.progress());
				DrawText("Entorno: Acrobot", static_cast<int>(envPanelBounds.x), 95, 20, DARKGRAY);
			} else if (loadedTaskCategory == 1) {
				mountainCarEnv.draw(envPanelBounds);
				DrawText("Entorno: Mountain Car", static_cast<int>(envPanelBounds.x), 95, 20, DARKGRAY);
			} else if (loadedTaskCategory == 2) {
				carEnv.draw(envPanelBounds);
				DrawText("Entorno: Racing Car", static_cast<int>(envPanelBounds.x), 95, 20, DARKGRAY);
				DrawText(TextFormat("Pasos: %d", carEnv.stepCount()),
					static_cast<int>(envPanelBounds.x), static_cast<int>(envPanelBounds.y + envPanelBounds.height - 20), 18, DARKGRAY);
			} else {
				const char* placeholder = "Entorno aun no implementado para esta tarea";
				int placeholderWidth = MeasureText(placeholder, 20);
				DrawText(placeholder,
					static_cast<int>(envPanelBounds.x + envPanelBounds.width / 2.0f - placeholderWidth / 2.0f),
					static_cast<int>(envPanelBounds.y + envPanelBounds.height / 2.0f), 20, GRAY);
			}

			float dividerX = networkPanelBounds.x + networkPanelBounds.width + 20;
			DrawLineEx({ dividerX, networkPanelBounds.y }, { dividerX, networkPanelBounds.y + networkPanelBounds.height }, 1.0f, LIGHTGRAY);

			int nextSpeedLevel = (snnSpeedLevel + 1) % SNN_SPEED_LEVEL_COUNT;
			DrawText(TextFormat("Velocidad: %s (ESPACIO para %s)",
					SNN_SPEED_LABELS[snnSpeedLevel], SNN_SPEED_LABELS[nextSpeedLevel]),
				20, 70, 20, DARKGRAY);
		} else if (menu.getStatus() == VS_AI_RACING_CAR) {
			aiCarEnv.drawTrack(racePanelBounds);
			humanCarEnv.drawCar(racePanelBounds, HUMAN_CAR_COLOR, false);
			aiCarEnv.drawCar(racePanelBounds, AI_CAR_COLOR, true);

			DrawText("Tú (rojo) vs IA (azul)", static_cast<int>(racePanelBounds.x), 90, 22, DARKGRAY);
			DrawText("Flechas o gamepad: arriba/abajo o gatillos acelerar/frenar, izquierda/derecha o stick izquierdo girar",
				static_cast<int>(racePanelBounds.x),
				static_cast<int>(racePanelBounds.y + racePanelBounds.height + 8), 18, GRAY);
		} else if (menu.getStatus() == VS_AI_ACROBOT) {
			// Same start/mid/end sub-step interpolation as "Evaluar red": animate each
			// pendulum smoothly across the just-taken step's motion for the whole interval
			// leading up to the next one, instead of snapping straight to the new pose.
			double acrobotProgress = raceDiscreteAccumulatorMs / realStepMsForTask(0);
			humanAcrobotEnv.draw(discreteLeftPanelBounds, acrobotProgress);
			aiAcrobotEnv.draw(discreteRightPanelBounds, acrobotProgress);
			DrawText("Tu", static_cast<int>(discreteLeftPanelBounds.x), 95, 20, DARKGRAY);
			DrawText("IA", static_cast<int>(discreteRightPanelBounds.x), 95, 20, DARKGRAY);
			float acrobotDividerX = discreteLeftPanelBounds.x + discreteLeftPanelBounds.width + 20;
			DrawLineEx({ acrobotDividerX, discreteLeftPanelBounds.y }, { acrobotDividerX, discreteLeftPanelBounds.y + discreteLeftPanelBounds.height }, 1.0f, LIGHTGRAY);
			DrawText("Manten presionado Izquierda o Derecha para aplicar torque",
				static_cast<int>(discreteLeftPanelBounds.x), static_cast<int>(discreteLeftPanelBounds.y + discreteLeftPanelBounds.height + 12), 18, GRAY);
		} else if (menu.getStatus() == VS_AI_MOUNTAIN_CAR) {
			humanMountainCarEnv.draw(discreteLeftPanelBounds);
			aiMountainCarEnv.draw(discreteRightPanelBounds);
			DrawText("Tu", static_cast<int>(discreteLeftPanelBounds.x), 95, 20, DARKGRAY);
			DrawText("IA", static_cast<int>(discreteRightPanelBounds.x), 95, 20, DARKGRAY);
			float mcDividerX = discreteLeftPanelBounds.x + discreteLeftPanelBounds.width + 20;
			DrawLineEx({ mcDividerX, discreteLeftPanelBounds.y }, { mcDividerX, discreteLeftPanelBounds.y + discreteLeftPanelBounds.height }, 1.0f, LIGHTGRAY);
			DrawText("Manten presionado Izquierda o Derecha para empujar el auto",
				static_cast<int>(discreteLeftPanelBounds.x), static_cast<int>(discreteLeftPanelBounds.y + discreteLeftPanelBounds.height + 12), 18, GRAY);
		} else if (menu.getStatus() == LOAD_NETWORK_FILE_MENU) {
			const std::string& taskLabel = snnTaskCategories()[static_cast<size_t>(selectedTaskCategory)].label;
			std::string title = "Modelos disponibles: " + taskLabel;
			int titleWidth = MeasureText(title.c_str(), 24);
			DrawText(title.c_str(), SCREEN_WIDTH / 2 - titleWidth / 2, 80, 24, DARKGRAY);

			if (modelButtons.empty()) {
				const char* empty = "No hay modelos guardados para esta tarea todavia.";
				int emptyWidth = MeasureText(empty, 20);
				DrawText(empty, SCREEN_WIDTH / 2 - emptyWidth / 2, SCREEN_HEIGHT / 2, 20, GRAY);
			}
			for (Button& button : modelButtons) button.draw(mousePosition);
		}
		menu.draw(mousePosition);
		if (menu.getStatus() == VS_AI_ACROBOT || menu.getStatus() == VS_AI_MOUNTAIN_CAR) {
			drawArrowIcon(menu.getLeftActionButton().getButton(), false, DARKGRAY);
			drawArrowIcon(menu.getRightActionButton().getButton(), true, DARKGRAY);
		}
		DrawSponsorLogos(SCREEN_WIDTH, SCREEN_HEIGHT);
		EndDrawing();
	}

	CloseWindow();

	return 0;
}
