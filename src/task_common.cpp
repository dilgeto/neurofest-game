#include "../include/task_common.hpp"

#include <algorithm>
#include <cmath>

double realStepMsForTask(int taskCategory) {
    switch (taskCategory) {
        case 0: return 200.0;           // Acrobot: DT = 0.2 s (acrobot_env.hpp)
        case 2: return 10.0;            // Racing Car: DT = 0.01 s (car_env.hpp)
        default: return 1000.0 / 30.0;  // Mountain Car: 30 fps convention
    }
}

std::vector<double> encodeCarObservation(const std::array<double, 9>& obs) {
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

double decodeCarContinuousAction(const SnnNetwork& net, int outputIndex) {
    double t = net.firstSpikeTimeForOutput(outputIndex);
    double value = (t < 0.0) ? 0.0 : (1.0 - t / SnnNetwork::SIM_WINDOW_MS);
    return std::clamp(value * 2.0 - 1.0, -1.0, 1.0);
}
