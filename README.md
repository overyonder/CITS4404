# Pong Neuroevolution Benchmark Suite

This project implements a neural network-driven Pong game with evolutionary training, in both Rust and C++. It is designed for benchmarking, teaching, and comparing evolutionary algorithms and neural network representations across languages and architectures.

## Project Structure

- `rust/` — Rust implementation (CLI & TUI, multiple engines)
- `C++/` — C++ implementation (console)
- `benchmark.sh` — Script to benchmark all engines with `hyperfine`

## Overview

Both implementations evolve neural networks to control Pong paddles, optimizing for skillful play using a genetic algorithm. The neural nets are 8-16-4-1 MLPs, taking game state as input and outputting paddle movement.

### Key Components
- **Game Logic:** Pong simulation with fair physics, paddle/ball collisions, and max velocities.
- **Neural Network:** 8 inputs (positions, velocities), 16/4 hidden, 1 output (move command).
- **Genetic Algorithm:** Population evolves via selection, crossover, and mutation.
- **Engines:**
  - **Rust:** Stack, Heap, SIMD, Concurrent (Rayon), GPU (WGSL)
  - **C++:** Single representation (doubles, heap)

## Evolutionary Algorithm (General)

1. **Initialization:**
   - Randomly initialize a population of neural net weights.
2. **Evaluation:**
   - Each individual is evaluated by playing games against others (round-robin tournament).
   - **Fitness:**
     - **Rust:** Number of successful paddle returns.
     - **C++:** Returns + shots + wins (see below).
3. **Selection:**
   - Top individuals (elites) are selected based on fitness.
4. **Crossover:**
   - New individuals are created by mixing genes from two parents.
     - **Rust:** Standard GA crossover (see code).
     - **C++:** Random gene-wise selection.
5. **Mutation:**
   - Randomly perturb weights to introduce variation.
6. **Replacement:**
   - Next generation is filled with elites, crossovers, and mutations.
7. **Repeat:**
   - For a fixed number of generations.

## Parameters & Effects
- **Population Size:** Number of individuals per generation (default 128).
- **Generations:** Number of evolutionary cycles.
- **Mutation Rate/Strength:** Higher values increase diversity but can destabilize learning.
- **Elite Count:** Number of top individuals preserved each generation.
- **Engine:** (Rust) Controls memory layout and computation (Stack/Heap/SIMD/Concurrent/GPU).

## C++ vs Rust: Key Differences

| Feature           | Rust                                    | C++                      |
|-------------------|-----------------------------------------|--------------------------|
| Data Type         | `f32` (float)                           | `double`                 |
| Fitness Function  | Returns (paddle-ball hits)              | Returns + shots + wins   |
| Crossover         | Standard GA crossover                   | Random gene-wise         |
| Mutation          | Standard GA mutation                    | Random gene + noise      |
| Parallelism       | Rayon/Threads/GPU (configurable)        | Single-threaded          |
| Engines           | Stack, Heap, SIMD, Concurrent, GPU      | Heap (single)            |
| Tournament        | Full round-robin                        | Full round-robin         |
| Save/Load         | Binary file (`.net`)                    | Log file (`fittest.log`) |

### Detailed C++ Fitness
- Combines: number of returns, number of shots (opponent forced to move), and wins.
- Differs from Rust, which currently only counts returns.

# Neural Network Pong Evolution (Rust vs C++)

## Project Overview
This project implements an **evolutionary neural network controller** for Pong paddles, using a genetic algorithm to optimize a fixed-topology neural net. Both Rust and C++ implementations are provided for benchmarking and educational comparison.

- **Game Logic:** Pong simulation with ball and paddle physics. The ball bounces off the ceiling/floor, and the paddle imparts velocity to the ball. Paddle and ball velocities are capped.
- **Neural Network:** 8-16-4-1 (input-hidden-hidden-output) structure. Inputs: paddle/ball positions and velocities. Output: paddle movement command.
- **Evolutionary Algorithm:** Population-based genetic algorithm with selection, crossover, mutation, and elitism. Fitness = number of successful returns (paddle hits).
- **Engines:** Multiple Rust engines (Stack, Heap, SIMD, Concurrent, GPU) allow for memory and performance experiments. C++ uses a single heap-based engine.

## Functional Specification Checklist
- [x] Dual-mode CLI/TUI (selectable from command line or interactive menu)
- [x] TUI allows engine/activation selection, training/simulation switching
- [x] Adjustable evolutionary parameters (generations, population, mutation, elite count)
- [x] Training dashboard with badges, sparklines, progress bars, and genome visualization
- [x] All engine types implemented and selectable
- [x] Neural network structure: 8-16-4-1, correct input mapping
- [x] Fitness = number of successful returns (paddle hits)
- [x] Physics: ball bounces, paddle/ball velocity limits
- [x] Benchmark script for fair Rust/C++ comparison (10 generations)
- [x] Comprehensive rustdoc comments and README documentation

## Game Logic & Evolution
- **Inputs:** Paddle positions/velocities, ball position/velocity (8 total)
- **Network:** 8 inputs → 16 hidden → 4 hidden → 1 output
- **Fitness:** Number of times the paddle returns the ball (forces opponent to move)
- **Evolution:**
    - Population initialized randomly
    - Elites survive, others are generated via crossover/mutation
    - Parameters: generations, population size, elite count, mutation rate/strength
    - Engine/activation selectable from TUI/CLI

## Usage

### Running the Rust Implementation
- **CLI:**
  ```sh
  cd rust
  cargo run --release -- --nogui --engine stack --generations 10
  ```
- **TUI:**
  ```sh
  cargo run --release
  ```
  (Select engine, activation, and parameters interactively)

### Running the C++ Implementation
```sh
cd C++
make # or ./build.sh if present
./pong_evolution 10   # Number of generations (default is 100 if omitted)
```

### Benchmarking Both Implementations
```sh
./benchmark.sh
```
- Benchmarks all Rust engines and the C++ implementation for 10 generations using `hyperfine`.
- Ensure both Rust and C++ binaries are built first.

### Generating and Browsing Rust Documentation
```sh
cd rust
cargo doc --open
```
- Opens the full rustdoc documentation in your browser.

### Cross-Evaluation and Format Compatibility
- C++ saves the best genome as `fittest.log` (custom format).
- Rust saves the best genome as a binary `.net` file.
- **Currently not compatible.**
- **Long-term:** Write a parser/converter for cross-evaluation.

## Settings Explained
- **generations:** Number of generations to evolve
- **population_size:** Number of individuals per generation
- **elite_count:** Number of elites preserved each generation
- **mutation_rate:** Probability of mutating a weight
- **mutation_strength:** Magnitude of mutation
- **engine:** (Rust) Select backend (Stack, Heap, SIMD, Concurrent, GPU)
- **activation:** Activation function (Tanh, Relu, Atan, Linear)

## Documentation
- All major algorithms, data structures, and design decisions are documented with rustdoc comments.
- See source files for step-by-step explanations, memory layout, and idiomatic Rust patterns.
- To view: `cargo doc --open` in the `rust` directory.

---

For detailed differences, see code comments and the C++/Rust comparison table above.
