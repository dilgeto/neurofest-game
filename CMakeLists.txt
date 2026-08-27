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

# Standalone demo: animates the 6 Izhikevich neuron behaviors under a shared, adjustable
# DC current (see src/main_izhikevich.cpp). Reuses button.cpp and snn_network.cpp's
# neuron-parameter table, but is otherwise independent of the NeuroGame app/menu.
add_executable(IzhikevichDemo
    src/main_izhikevich.cpp
    src/button.cpp
    src/snn_network.cpp
    src/branding.cpp
)

target_include_directories(IzhikevichDemo PRIVATE include)
target_link_libraries(IzhikevichDemo PRIVATE raylib)

# Standalone demo: shows spike transport and postsynaptic voltage for one excitatory and
# one inhibitory neuron pair (see src/main_synapse_demo.cpp). Reuses snn_network.cpp's
# neuron-parameter table and button.cpp for the separated/merged view toggle.
add_executable(SynapseDemo
    src/main_synapse_demo.cpp
    src/button.cpp
    src/snn_network.cpp
    src/branding.cpp
)

target_include_directories(SynapseDemo PRIVATE include)
target_link_libraries(SynapseDemo PRIVATE raylib)

# Standalone demo: shows how 4 spike encoders (TTFS linear/logarithmic, Rate, Poisson)
# turn a single number into a spike train (see src/main_encoder_demo.cpp). Formulas are
# reimplemented locally (verified against snn-simulator/src/encoding/*), so this target
# has no dependency on the rest of NeuroGame beyond raylib itself.
add_executable(EncoderDemo
    src/main_encoder_demo.cpp
    src/branding.cpp
)

target_include_directories(EncoderDemo PRIVATE include)
target_link_libraries(EncoderDemo PRIVATE raylib)
