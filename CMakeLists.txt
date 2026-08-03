cmake_minimum_required(VERSION 3.15)
project(NeuroGame CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(raylib REQUIRED)

add_executable(NeuroGame
    src/main.cpp
    src/menu.cpp
    src/button.cpp
    src/snn_network.cpp
    src/model_browser.cpp
    src/acrobot_env.cpp
    src/mountain_car_env.cpp
    src/car_env.cpp
)

target_include_directories(NeuroGame PRIVATE include)
target_link_libraries(NeuroGame PRIVATE raylib)
