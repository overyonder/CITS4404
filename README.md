# Pong Neuroevolution Benchmark Suite

This project provides Rust and C++ implementations of a neural network-driven Pong game, designed for benchmarking, teaching, and comparing evolutionary algorithms across different languages and hardware architectures.

## ✨ Features

- **Dual Implementations:** Full-featured Rust application and a legacy C++ version for comparison.
- **CLI & TUI Modes (Rust):** Run training sessions from the command line or use the interactive Terminal UI.
- **Multiple Compute Engines (Rust):** Choose from different neural network engines, each with unique performance and memory trade-offs:
  - **`Stack`**: Fixed-size, stack-allocated arrays for maximum performance with zero allocations.
  - **`Heap`**: Dynamically-sized, heap-allocated vectors for greater flexibility.
  - **`SIMD`**: Accelerated with architecture-specific intrinsics (AVX2/FMA) for faster math.
  - **`GPU`**: Massively parallel computation using `wgpu` and custom WGSL shaders.
- **Concurrent Mode:** Host-level concurrency (`stack`, `heap`, `simd`) can be enabled with the `--concurrent` flag to leverage multiple CPU cores.
- **Genetic Algorithm:** Evolves neural network weights to control Pong paddles, optimizing for skillful play.
- **Comprehensive Benchmarking:** A robust `benchmark.sh` script using `hyperfine` to compare all Rust engines and the C++ version.
- **Extensive Documentation:** In-code `rustdoc` comments explain algorithms, data structures, and design patterns.

## 🚀 Getting Started

### Prerequisites

- **Rust:** `cargo`
- **C++:** `g++` and `make`
- **Benchmarking:** `hyperfine`

### 🦀 Rust Version (CLI & TUI)

1.  **Navigate to the Rust directory:**
    ```sh
    cd rust
    ```

2.  **Run the interactive TUI:**
    ```sh
    cargo run --release
    ```
    Inside the TUI, you can select the engine, activation function, and run training or simulation sessions.

3.  **Run directly from the CLI:**
    ```sh
    # Example: Run 10 generations with the SIMD engine
    cargo run --release -- --nogui --engine simd --generations 10
    ```

###  C++ Version

1.  **Navigate to the C++ directory:**
    ```sh
    cd C++
    ```
2.  **Build the project:**
    ```sh
    make
    ```
3.  **Run the simulation:**
    ```sh
    # Run for 10 generations
    ./pong_evolution 10
    ```

## 📊 Benchmarking

The `benchmark.sh` script provides a standardized way to compare the performance of all available engines.

**To run the benchmarks:**
```sh
./benchmark.sh
```

The script will:
1.  Build the Rust project in release mode.
2.  Build the C++ project using its `Makefile`.
3.  Run `hyperfine` on all Rust engines in both single-threaded and concurrent modes.
4.  Run `hyperfine` on the C++ implementation for comparison.

## 🧠 Evolutionary Algorithm

The core of this project is a genetic algorithm that trains a population of neural networks.

-   **Neural Network:** A fixed-topology MLP with an **8-16-4-1** structure.
    -   **Inputs (8):** Ball and paddle positions/velocities, all normalized.
    -   **Hidden Layers (16, 4):** Two hidden layers with a configurable activation function.
    -   **Output (1):** A single value determining the paddle's upward or downward movement.
-   **Fitness Function:** Fitness can be measured in several ways, configurable via the `--fitness-func` flag. The default (`CppEquivalent`) measures successful ball returns, while others (`Balanced`, `Performance`) incorporate rally length and game duration to reward more complex behaviors.
-   **Evolution Process:**
    1.  **Selection:** The best-performing individuals (elites) are preserved.
    2.  **Crossover:** New individuals are created by combining the weights of two elite parents.
    3.  **Mutation:** Small, random changes are introduced into the weights of new individuals to foster diversity.

## ⚙️ Configuration

You can configure the training process via CLI flags or the TUI:

-   `--engine`: The Rust engine to use (`stack`, `heap`, `simd`, `gpu`).
-   `--concurrent`: Use host-level concurrency (Rayon) for evaluation. Works with `stack`, `heap`, and `simd` engines.
-   `--activation`: The activation function for hidden layers (`tanh`, `relu`, `atan`, `linear`, `sigmoid`).
-   `--generations`: The number of generations to run.
-   `--population-size`: The number of individuals in each generation.
-   `--mutation-rate`: The probability (0.0 to 1.0) that a weight will be mutated.
-   `--mutation-strength`: The magnitude of the random change applied during mutation.
-   `--elite-count`: The number of top individuals to carry over to the next generation.

## 📚 Documentation

This project is heavily documented for teaching purposes. To explore the full `rustdoc` documentation:

1.  **Navigate to the Rust directory:**
    ```sh
    cd rust
    ```
2.  **Generate and open the docs:**
    ```sh
    cargo doc --open
    ```

## 🚧 C++ Compatibility

The Rust and C++ versions save their best-performing neural networks in different formats (`.net` for Rust, `fittest.log` for C++). A compatibility layer or parser would be required for cross-evaluation, which is a potential area for future work.
