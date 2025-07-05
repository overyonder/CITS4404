# Neural Network Rust Application Review & Documentation Plan

## Overview
This plan outlines the process for reviewing, documenting, and improving a Rust-based neural network tool for Pong, which supports both command line (CLI) and terminal user interface (TUI) modes. The application leverages a genetic algorithm to optimize neural networks that control the paddles in a Pong game, with several engine types for benchmarking and performance comparison. The project also includes a legacy C++ implementation for reference and benchmarking.

## Application Logic
- **Modes:**
  - CLI mode: Command-line operation for automation and scripting.
  - TUI mode: Interactive terminal interface allowing users to select actions (train, simulate, visualize, etc.).
- **Training:**
  - Uses a genetic algorithm with tunable parameters (e.g., mutation rate, crossover, elite count).
  - Dashboard displays real-time training statistics (best genome, fitness, etc.).
  - On completion, the best genome is displayed and/or saved to a binary file for later use.
- **Simulation:**
  - Supports running simulations with selectable left/right paddle engines (Rust or C++).
  - Compatibility layer required for converting C++ output to Rust format.

## Neural Network Engines
- **Stack:** Fixed-size array genomes (stack-allocated).
- **Heap:** Vector genomes (heap-allocated, fixed size).
- **SIMD:** Uses SIMD instructions for performance.
- **Concurrent:** Parallelized versions of the above using Rayon.
- **GPU:** Uses WGSL for GPU-accelerated tournaments.

## Fitness Function
- Based on paddle-ball interactions: counts how often a paddle hits the ball in a way that forces the opponent to move, rather than just score.
- Ball and paddle physics:
  - Ball bounces off ceiling/floor.
  - Paddle imparts its velocity to the ball.
  - Maximum velocities enforced for both ball and paddle.

## Documentation & Comments
- Every Rust file should include thorough rustdoc comments:
  - Outline the logic of each algorithm step.
  - Explain all data structures and their memory allocation (stack, heap, cache optimization, etc.).
  - Describe the effects and idiomatic patterns of evolutionary algorithm parameters (crossover, mutation, elite, etc.).
- All documentation should be suitable for teaching a computer science class.

## README.md Updates
- Concise, logical summary of the application, its parts, and algorithms.
- Detailed description of settings/options (e.g., mutation rate effects).
- Comparison with the legacy C++ version (`C++/Evolve.cpp`).
- Benchmarking process using `benchmark.sh`.
- Paddle selection and compatibility notes.

## Compatibility Layer
- Write a converter for `fittest.log` (C++) to Rust binary weights.
- Ensure parameter compatibility between C++ and Rust implementations.

## Error Correction & Optimization
- As each file is reviewed, correct small errors or inconsistencies.
- Optimize structure or clarity where possible.

## TUI & ratatui Enhancements
- Leverage advanced ratatui features (canvas, badges, sparklines, tournament bracket, progress bars, etc.), inspired by `ratatui/examples/demo`.
- All UI code must be connected to real backend data/methods (no mocked or static displays).
- Add detailed, teaching-focused comments to all ratatui code sections.

## Task List
1. Review each Rust source file for correctness and completeness.
2. Add/expand rustdoc comments for every file and data structure.
3. Document algorithm steps and memory allocation in comments.
4. Update README.md as described above.
5. Implement compatibility layer for C++ net weights in Rust.
6. Correct any found errors or improve structure/clarity as needed.
7. Enhance TUI with advanced ratatui features and ensure all UI reflects real backend data.
8. Add detailed, teaching-focused comments to all ratatui code sections.

## Current Goal
- Begin by reviewing and documenting each Rust source file, following the above guidelines.
