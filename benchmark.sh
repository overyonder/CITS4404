#!/bin/bash

# Benchmark script for Rust and C++ Pong Neuroevolution Engines
# Requires: hyperfine, cargo, g++, make

set -e

# --- Configuration ---
GENERATIONS=10
RUST_BIN="rust/target/release/pong"
CPP_DIR="C++"
CPP_BIN_NAME="pong_evolution"
CPP_BIN_PATH="$CPP_DIR/$CPP_BIN_NAME"

# --- Rust Build & Benchmarks ---
echo "--- Building Rust Project ---"
(cd rust && cargo build --release)
echo ""

# Engines to benchmark (match CLI args)
# Added 'concurrent' to the list
ENGINES=(stack simd heap gpu concurrent)

echo "--- Benchmarking Rust Engines (Generations: $GENERATIONS) ---"
for engine in "${ENGINES[@]}"; do
    echo "Benchmarking Rust engine: $engine"
    hyperfine -w 2 -r 10 "$RUST_BIN --nogui --engine $engine --generations $GENERATIONS"
done
echo ""


# --- C++ Build & Benchmark ---
echo "--- Building C++ Project ---"
if [ -f "$CPP_DIR/Makefile" ]; then
    echo "Building C++ project with make..."
    make -C "$CPP_DIR"

elif [ -f "$CPP_DIR/build.sh" ]; then
    echo "Building C++ project with build.sh..."
    (cd "$CPP_DIR" && ./build.sh)
else
    echo "No build script or Makefile found for C++ project. Please build manually."
    # Exit gracefully if no build method is found
    exit 0
fi
echo ""

if [ ! -f "$CPP_BIN_PATH" ]; then
    echo "C++ binary not found at $CPP_BIN_PATH after build attempt."
    exit 1
fi

# Benchmark C++ implementation
echo "--- Benchmarking C++ Implementation (Generations: $GENERATIONS) ---"
hyperfine -w 2 -r 10 "$CPP_BIN_PATH $GENERATIONS"

# Note: The C++ output net format (fittest.log) is not directly compatible with Rust,
# but a parser could be written to convert it for cross-evaluation in the future.

echo ""
echo "Benchmarking complete."
