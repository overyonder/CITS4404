---
trigger: always_on
description: When updating the master plan, make the corresponding changes to the requirements (e.g. add/remove items, tick/untick items, etc.)
---

## Part 2: Project requirements

### Neural Network Pong Application

#### Notes

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

The CLI is to be well documented with --help.

The TUI is to be feature rich including features like ratatui's widgets (chart / sparkline / bar graph / lots of colors, etc.). There could be information about generations (current / total), number of games completed as a progress bar / line gauge. We should show the current engine in use with a colourful badge in the border (e.g. stack = grey, SIMD = orange, concurrent = teal, GPU = purple) etc.
We should show sparklines for CPU / GPU usage, a stopwatch for the current run. We should show the current overall best genome as an encoding in a bar chart (217 x axis bars, height corresponds to weight) or a colour chart.

The documentation of new code should follow the same pattern as the existing files, which are heavily documented with teaching notes, as this is a computer science teaching project.

#### Task List

- [ ] Confirm that each engine actually does what it is intended to (e.g. does the stack implementation use the stack for forward passes, or is it likely to end up heap allocated if it's part of a larger heap allocated struc?)
- [ ] Confirm that each parameter actually does as intended too, i.e.
  - [ ] Do each of the activation functions work as they describe? Does one of them match the C++ implementation version?
  - [ ] Does the tournament work as described - round robin? Does this match the C++ implementation version of competition?
  - [ ] Do each of the fitness evaluation functions work as described (e.g. the C++ default, the optimized version, and the paramaterized version?)
- [ ] Confirm the compatibility layer for running a simulation using the C++ weights (in fittest.log) works using the conversion as designed. Write a test for this that processes a known fittest.log and compares the result.
- [ ] Is everything making the best use of the logging framework to print info and debug logs at appropriate locations?
- [ ] Identify missing or incomplete features

Currently the training dashboard shows:

- Progress (gauge)
- Fitness history (blocks)
- Champion genome (not updating currently)
- Info (single line of text)
- Log (not updating currently)

The following fixes are required:

- [ ] Fitness history should use a bar graph instead of blocks, per the example in examples/barchart.rs. It must also show the actual fitness number so we know the relative strength as generations progress.
- [ ] The blocks for each section should use nice borders with BorderType::Rounded rather than dash characters.
- [ ] The simulation UI seems to use canvas which is correct. Please compare this to the example in examples/canvas.rs for any ideas on making the drawing more idiomatic for ratatui, if needed.
- [ ] The percentage progress gauge is helpful but under it, we should have a frame with text for the number of generations done / remaining / currently processing. Under this, we should have blocks showing all of the generations/matchups to be completed (toggleable with tabs per examples/tabs.rs). These should be grey blocks, similar to a defragmenter UI. When the generation/matchup is complete, it should turn green. When it is being executed (e.g. currently running on the relevant engine) it should turn orange. This means that multiple blocks might be orange in concurrent mode.
- [ ] The champion genome should be shown as 7 x 31 blocks (it is 217 parameters). The colour of each block should depend on the intensity of that weight (since each is a 32 bit float, this should be able to be normalized to an RGB value). Refer to examples/colors.rs for an example.
- [ ] The log should actually log the messages from tracing. Currently it only shows "starting training". Refer to examples/list.rs for an example of how to have a scrolling log window in ratatui.
- [ ] The info panel should show colorful badges in the corner to indicate which engine (Stack, SIMD, GPU) is currently in use. (E.g. GPU = purple, stack = orange, heap = red, SIMD = blue, etc.). It should also show a badge for the fitness function, activation function, and whether or not it is in concurrent mode. Other parameters like the population size, mutation rate, and mutation strength should also be listed.

- [ ] Write idiomatic rust tests for `cargo test` for each testable major feature of the application to enable early identification of breaking changes as new features are added in the future

#### Future ideas

2. Advanced Genetic Algorithm Concepts
   To deepen the educational value, we could introduce more GA techniques and make them configurable.
   Alternative Selection Methods: Implement other classic selection strategies like Tournament Selection or Roulette Wheel Selection and allow the user to choose between them via a CLI flag. This would be an excellent way to compare their effects on evolution.
   Configurable Crossover/Mutation: Add different types of crossover (e.g., single-point) and mutation (e.g., Gaussian noise) to demonstrate how different genetic operators can affect the search for a solution.

3. Deeper Analysis and Usability
   Checkpointing: For very long training runs (1000+ generations), the ability to save the entire population's state every N generations and resume later would be invaluable. As a temporary solution, the ability to seed the initial population with an existing best_model.bin would help.
   TUI-based Benchmarking: Create a TUI screen that automates the benchmarking process, running all engines for a set number of generations and presenting the results (time per generation, final fitness) in a clean table.
