#pragma once
#include <array>
#include <vector>

#include "car_env.hpp"
#include "snn_network.hpp"

// Small helpers shared by NetworkViewModule and VsAiRacingCarModule (extracted from
// main.cpp, where they back "Evaluar red" and "VS IA: Racing Car" respectively).

// Wall-clock duration of one real env step, per task -- what "Real" playback speed maps the
// 20ms SNN decision window to. Acrobot and Racing Car have an exact physical DT (ported
// straight from rl-tools' environment parameters); Mountain Car's rl-tools step is
// dimensionless (no physical DT), so it falls back to this project's own real-time playback
// convention for it (wann-cpp/replay_mountain_car.py defaults to --fps 30).
double realStepMsForTask(int taskCategory);

// Normalizes a raw CarEnv observation into [0,1] per channel, matching SnnCarTask.cpp's
// TTFS-path normalization exactly (position/heading/velocities scaled by their bounds; the 3
// lidar readings are already in [0,1]).
std::vector<double> encodeCarObservation(const std::array<double, 9>& obs);

// RLDecoder::FIRST_SPIKE continuous decode (rlDecoder.cpp's single-arg overload, as used by
// SnnCarTask.cpp's decodeActions): earlier spike -> value closer to +1; no spike -> 0 before
// the *2-1 remap, i.e. -1 after it.
double decodeCarContinuousAction(const SnnNetwork& net, int outputIndex);
