# AI
A attempt to make a AI in 48 hours zero to hero

not to mention you will need to have mnist_train.csv for the second neural network file(preferably from kaggle a place where you can find all sorts of files

## Prerequisites

- C++17 compatible compiler (g++ or clang++)
- SDL2 library
- CMake (optional, but recommended)



To run the cpp file do

    g++ -std=c++17 digit_drawer.cpp -o digit_drawer `sdl2-config --cflags --libs`

This command is for linux and mack idk about windows

then run...

    ./digit_drawer
### Linux (Debian/Ubuntu)


    sudo apt-get install g++ libsdl2-dev cmake


### macOS (using Homebrew)

    brew install gcc sdl2 cmake

### Windows (MSYS2)

    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-SDL2 mingw-w64-x86_64-cmake

## Building the Project
# Basic Build


    g++ -std=c++17 digit_drawer.cpp -o digit_drawer `sdl2-config --cflags --libs`

## Using CMake (Recommended)

# Create a build directory:


    mkdir build && cd build

Run CMake and build:


    cmake ..
    make

### Running the Application

    ./digit_drawer

Usage

    Drawing:

        Left-click and drag to draw digits

        The canvas is 280x280 pixels but downsamples to 28x28 for the neural network

    Recognition:

        The network predicts the digit in real-time

        Predictions are shown in the console output

    Controls:

        C key: Clear the canvas

        Close the window to exit
