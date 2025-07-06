---
trigger: always_on
description: When updating the master plan, make the corresponding changes to the requirements (e.g. add/remove items, tick/untick items, etc.)
---

## Part 2: Project requirements

### Neural Network Pong Application

#### Notes

Logic:
- The application is a neural network tool with a command line mode and a TUI mode.
- In TUI mode, the user can choose between training the neural net, visualizing a simulation, etc.
- Training uses a genetic algorithm with initial parameters that can be adjusted from the TUI.
- When running training, there is a dashboard showing information about the training run.
- When it is complete, the best genome found is saved to a binary file (so it can later be loaded and shown running in a simulation).

The genetic algorithm is optimizing parameters to a neural network that controls the pong paddles in the game.
There are several different representations of the neural network which can be tested (for benchmarking). All are 8-16-4-1 networks (the inputs come from the current game state, i.e. both paddles y position and velocity, since they are fixed at the walls in x, and the ball's x and y position and velocity).
The representations (called engines) are:
- Stack (genomes are fixed size arrays)
- Heap (genomes are vectors, although size doesn't change)
- SIMD (uses SIMD instructions)
- Concurrent (exists for the above representations, uses rayon to process multiple games simultaneously)
- GPU (uses WGSL to process the tournament on the GPU)

Fitness function selection is modular, with options for:
- The default function which matches the C++ implementation (for apples-apples comparison)
- One suggested by the LLM, which should be clearly documented
- A parameterized one configurable by factors (also documented)

Activation function for training is also selectable, and includes standard options like relu, sigmoid, etc., as well as the default which matches the C++ activation function.

When saving a neural network to a binary file, the options used to generate it (e.g. number of generations, activation function, fitness function, evolution parameters, etc.) are saved in the file as a comment header.

Some fundamentals about the game mechanics:
- The ball bounces fairly off the ceiling and floor
- The paddle imparts its velocity onto the ball after hitting it
- There is a maximum velocity for the ball and the paddle.

The CLI is to be well documented with --help.

The TUI is to be feature rich including features like ratatui's widgets (chart / sparkline / bar graph / lots of colors, etc.). There could be information about generations (current / total), number of games completed as a progress bar (it could show games in progress as green blocks, similar to a defrag viewer). We could show the current engine in use with a colourful badge in the border (e.g. stack = grey, SIMD = orange, concurrent = teal, GPU = purple) etc.
It could even show a tournament bracket each generation. We could show sparklines for CPU / GPU usage, a stopwatch for the current run, a line gauge for progress. We could show the current overall best genome as an encoding in a bar chart (217 x axis bars, height corresponds to weight) or a colour chart. Please have a go at making the best most feature rich version you can, making reference to the demo provided.

#### Task List

It is important to go through every rust file and include rustdoc comments for completeness, and make this all congruent with the detailed README.md. The comments should also specifically outline the logic of each numbered algorithm step, and explain the algorithm. Similarly, it should explain data structures, and how they are allocated in memory (stack / heap / optimized for caching or staying on registers, etc.). This is because this is going to be for teaching a computer science class. Also explain the effects of each of the parameters to the main evolutionary functions (e.g. crossover, mutation, elite, etc.) using idiomatic patterns for these data structures and algorithms in computer science. Much of the documentation is already complete, so we are just looking for any and all improvements that could be made.

If we find any small errors / inconsistencies / or opportunities to make the structure more optimized or clear, please correct these as we go, although hopefully this isn't necessary.

REMEMBER: update the plan as we go

- [ ] Review all Rust source files for completeness and correctness (main.rs, config.rs, constants.rs, gamestate.rs, population.rs, traits.rs)
- [ ] Review this plan and make items clearer / more accurate where necessary (e.g. remove duplicate tasks, order tasks logically)
  - [ ] Review all engine implementations (stack, heap, simd, gpu)
  - [ ] Review forward.wgsl shader for GPU engine
  - [ ] Review TUI core files (mod.rs, app.rs, ui.rs, training.rs, simulation.rs)
  - [ ] Review benchmarking script (benchmark.sh)
- [ ] Identify missing or incomplete features (especially TUI training/simulation/visualization)
- [ ] Check all game logic for compliance with described physical rules
- [ ] Ensure all engine types are benchmarkable
- [ ] Confirm thorough rustdoc comments (algorithms, data structures, memory allocation)
- [ ] Update README.md with structured documentation and comparison to C++
- [ ] Check compatibility layer for C++ weights in Rust
- [ ] Ensure all Activation enum variants are handled in all engines
- [ ] Check activation mapping between host and shader in GPU engine
- [ ] Check all engine support in TUI training mode
- [ ] Check TUI error messages persistent until dismissed
- [ ] Check CLI progress reporting (stdout per generation)
- [ ] Ensure CLI --help prints all parameters with defaults
- [ ] Check TUI configuration screen before training
- [ ] Check TUI training dashboard uses advanced features (progress bar, engine badge, sparklines, stopwatch, genome chart, etc.)
- [ ] Create rust tests for C++ compatibility layer
- [ ] Check each engine for possible optimizations (staying within their design intent, e.g. heap allocation for the heap engine).
- [ ] Check fitness function is modular/selectable (C++-equivalent, current, improved)
  - [ ] Check C++-equivalent fitness function is implemented with matching logic
  - [ ] Check current Rust fitness function is a selectable option
  - [ ] Check design and implementation of third, improved fitness function (e.g., address "quick win" reward issue where a better individual might get a lower score due to having less returns since it's already beat the enemy)
  - [ ] Check CLI/TUI option to select fitness function
  - [ ] Check parameterization of fitness function (weights for returns, time, shots, wins, etc.) working correctly
  - [ ] Update documentation and help text for fitness selection
- [ ] Write idiomatic rust tests for each other major feature of the application to enable early identification of breaking changes as new features are added in the future
