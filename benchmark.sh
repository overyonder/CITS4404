#!/bin/bash

# Benchmark script for Rust and C++ Pong Neuroevolution Engines
# Requires: hyperfine, cargo, g++, make

set -e

# Rust build and benchmarks
cd rust

echo "Building Rust project..."
cargo build --release

# Engines to benchmark (match CLI args)
ENGINES=(stack simd heap gpu)

for engine in "${ENGINES[@]}"; do
    echo "Benchmarking Rust engine: $engine"
    hyperfine -w 2 -r 10 "target/release/pong --nogui --engine $engine --generations 10"
done

cd ..

# C++ build and benchmark
cd "C++"

if [ -f Makefile ]; then
    echo "Building C++ project with make..."
    make
elif [ -f build.sh ]; then
    echo "Building C++ project with build.sh..."
    ./build.sh
else
    echo "No build script found for C++ project. Please build manually."
fi

CPP_BIN="./pong"
if [ ! -f "$CPP_BIN" ]; then
    echo "C++ binary not found at $CPP_BIN. Please adjust the script."
    exit 1
fi

# Benchmark C++ implementation
# The C++ version accepts the number of generations as an argument (default 100)
# For fair benchmarking, we use 10 generations (like Rust)
echo "Benchmarking C++ implementation..."
if [ -f "C++/pong_evolution" ]; then
    hyperfine --warmup 1 "C++/pong_evolution 10"
else
    echo "C++ binary not found. Please build C++/pong_evolution manually."
fi

# Note: The C++ output net format (fittest.log) is not directly compatible with Rust, but a parser could be written to convert it for cross-evaluation in the future.

cd ..

echo "Benchmarking complete."
