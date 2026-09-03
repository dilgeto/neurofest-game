cmake_minimum_required(VERSION 3.15)
project(NeuroGame CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(raylib)

add_executable(NeuroGame
    src/main.cpp
    src/menu.cpp
    src/gamepad_setup.cpp
    src/button.cpp
    src/snn_network.cpp
    src/model_browser.cpp
    src/acrobot_env.cpp
    src/mountain_car_env.cpp
    src/car_env.cpp
    src/iris_env.cpp
    src/branding.cpp
)

target_include_directories(NeuroGame PRIVATE include)
target_link_libraries(NeuroGame PRIVATE raylib)

# Standalone demo: animates the 6 Izhikevich neuron behaviors under a shared, adjustable DC
# current. main_izhikevich.cpp is now a thin wrapper around izhikevich_module.cpp, which
# MultiDemo also links against.
add_executable(IzhikevichDemo
    src/main_izhikevich.cpp
    src/izhikevich_module.cpp
    src/button.cpp
    src/snn_network.cpp
    src/branding.cpp
)

target_include_directories(IzhikevichDemo PRIVATE include)
target_link_libraries(IzhikevichDemo PRIVATE raylib)

# Standalone demo: shows spike transport and postsynaptic voltage for one excitatory and one
# inhibitory neuron pair. main_synapse_demo.cpp is now a thin wrapper around
# synapse_module.cpp, which MultiDemo also links against.
add_executable(SynapseDemo
    src/main_synapse_demo.cpp
    src/synapse_module.cpp
    src/button.cpp
    src/snn_network.cpp
    src/branding.cpp
)

target_include_directories(SynapseDemo PRIVATE include)
target_link_libraries(SynapseDemo PRIVATE raylib)

# Standalone demo: shows how 2 spike encoders (TTFS linear, Rate) turn a single number into a
# spike train. Formulas are reimplemented locally (verified against
# snn-simulator/src/encoding/*). main_encoder_demo.cpp is now a thin wrapper around
# encoder_module.cpp, which MultiDemo also links against.
add_executable(EncoderDemo
    src/main_encoder_demo.cpp
    src/encoder_module.cpp
    src/branding.cpp
)

target_include_directories(EncoderDemo PRIVATE include)
target_link_libraries(EncoderDemo PRIVATE raylib)

# Standalone demo: shows the two encoders actually used by this project's trained WANN
# models -- TTFS and SMALL/signed -- as opposed to EncoderDemo's generic textbook survey.
# main_encoder_demo_wann.cpp is now a thin wrapper around encoder_wann_module.cpp, which
# MultiDemo also links against.
add_executable(EncoderDemoWANN
    src/main_encoder_demo_wann.cpp
    src/encoder_wann_module.cpp
    src/snn_network.cpp
    src/branding.cpp
)

target_include_directories(EncoderDemoWANN PRIVATE include)
target_link_libraries(EncoderDemoWANN PRIVATE raylib)

# Standalone demo: shows how the FIRST_SPIKE and RATE_ARGMAX decoders (rlDecoder.cpp's
# RLDecoder::decodeDiscreteAction) pick a winner among several channels' spike trains, and
# how they can disagree given the exact same spikes. main_decoder_demo.cpp is now a thin
# wrapper around decoder_module.cpp, which MultiDemo also links against.
add_executable(DecoderDemo
    src/main_decoder_demo.cpp
    src/decoder_module.cpp
    src/branding.cpp
)

target_include_directories(DecoderDemo PRIVATE include)
target_link_libraries(DecoderDemo PRIVATE raylib)

# Standalone demo: runs a live WANN (Weight Agnostic Neural Network) evolution against a
# small 2D classification task and animates population, champion topology and fitness
# automatically. main_wann_evolution_demo.cpp is now a thin wrapper around
# wann_evolution_module.cpp, which MultiDemo also links against. Ind.cpp/Wann.cpp/
# NsgaSort.cpp are an unmodified vendored copy of the real algorithm from ../wann-cpp
# (include/wann/), minus the JSON hyperparameter loader (Hyperparams.h here is a trimmed
# struct-only copy) so this target has no dependency beyond raylib and the STL.
add_executable(WannEvolutionDemo
    src/main_wann_evolution_demo.cpp
    src/wann_evolution_module.cpp
    src/Ind.cpp
    src/Wann.cpp
    src/NsgaSort.cpp
    src/button.cpp
    src/branding.cpp
)

target_include_directories(WannEvolutionDemo PRIVATE include)
target_link_libraries(WannEvolutionDemo PRIVATE raylib)

# Combines 2 or 3 of NeuroGame/the standalone demos' modules into one window at once, stacked
# vertically, picked from a small selector screen (see src/main_multi_demo.cpp and
# include/demo_module.hpp). Links every demo's extracted module .cpp -- the same ones each
# standalone binary wraps, plus NeuroGame's "Evaluar red" (x3, one per task) and "VS IA" (x4)
# modules -- so there is exactly one copy of each demo's logic.
add_executable(MultiDemo
    src/main_multi_demo.cpp
    src/gamepad_setup.cpp
    src/izhikevich_module.cpp
    src/synapse_module.cpp
    src/encoder_module.cpp
    src/encoder_wann_module.cpp
    src/decoder_module.cpp
    src/network_view_module.cpp
    src/vs_ai_discrete_module.cpp
    src/vs_ai_racing_car_module.cpp
    src/vs_ai_iris_module.cpp
    src/wann_evolution_module.cpp
    src/Ind.cpp
    src/Wann.cpp
    src/NsgaSort.cpp
    src/task_common.cpp
    src/model_browser.cpp
    src/acrobot_env.cpp
    src/mountain_car_env.cpp
    src/car_env.cpp
    src/iris_env.cpp
    src/button.cpp
    src/snn_network.cpp
    src/branding.cpp
)

target_include_directories(MultiDemo PRIVATE include)
target_link_libraries(MultiDemo PRIVATE raylib)
