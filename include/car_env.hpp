#pragma once
#include <raylib.h>
#include <array>
#include <random>

// Ports rl-tools' CarTrack environment (rl_tools/rl/environments/car): a dynamic bicycle
// model with a simplified Pacejka tire model, driving on the hardcoded 100x100 track
// bitmap from rl_tools/rl/environments/car/track.h, with 3 lidar-style distance sensors.
// Matches wann-cpp's SnnCarTask.cpp (CarSpec: 100x100 grid, 50mm/px -> a 5x5m track).
class CarEnv {
    public:
        static constexpr double DT = 0.01;           // physics integration step [s]
        static constexpr double TRACK_SCALE = 0.05;  // [m] per grid cell
        static constexpr int TRACK_SIZE = 100;
        static constexpr double BOUND = TRACK_SCALE * TRACK_SIZE / 2.0; // 2.5 m
        static constexpr double VX_MAX = 3.0, VY_MAX = 2.0, OMEGA_MAX = 10.0;
        static constexpr int EPISODE_STEP_LIMIT = 1000;

        void reset(std::mt19937& rng);

        // Raw observation: [x, y, mu, vx, vy, omega, lidar_left, lidar_center, lidar_right]
        // (lidar_* already in [0,1]: fraction of the 1.25m max ray length).
        std::array<double, 9> observe() const;

        // Applies (throttleBrake, steering) in [-1,1] for DT seconds. Auto-resets on
        // leaving the track or hitting the step limit.
        void step(double throttleBrake, double steering, std::mt19937& rng);

        void draw(Rectangle bounds) const;

    private:
        bool onTrack(double worldX, double worldY) const;
        double lidarDistance(double directionOffset) const;

        double x_ = 0.0, y_ = 0.0, mu_ = 0.0;
        double vx_ = 0.0, vy_ = 0.0, omega_ = 0.0;
        int stepCount_ = 0;

        mutable Texture2D trackTexture_{};
        mutable bool trackTextureReady_ = false;
        void ensureTrackTexture() const;
};
