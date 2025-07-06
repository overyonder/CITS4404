Please proceed in line with the following task list:

#### Task List

- [ ] @gpu.rs#L268-269 check if the function should be changed to not need an argument, to avoid the need for cloning
- [ ] **Logical Bug:** Check if the tournament algorithm is testing matchups/genomes that have already happened.
- [ ] Confirm Rust implementation matches original C++ behaviour. Evolve.cpp contains the original code, this should be compared to the Rust implementation for the following: tournament matchup method, activation function, fitness evaluation, game logic, constants.
- [ ] **Save Model Option**: Save the model after each generation, with an option to **load an existing model** to seed the initial population for future runs.

- [ ] **TUI Improvements:**
  - [ ] Confirm that the scrolling log window is working
  - [ ] Confirm that messages from Vulkan / GPU are suppressed when in TUI mode to avoid pushing it off screen.

---

#### Future Features / Ideas

1. **Advanced Genetic Algorithm Concepts**

   - Alternative Selection Methods: Implement other selection tournament strategies like **Single Elimination**, **Double Elimination Selection** and **Roulette Wheel Selection** with CLI flags for easy comparison.
   - Configurable Crossover/Mutation: Add options for **different crossover types** (e.g., single-point) and **mutation types** (e.g., Gaussian noise). Parameters should be configurable via CLI or UI and saved in the binary file header.
   - **Crossover Rate**: Add parameter. Default set to **0.85**, configurable for future flexibility. Parameters for **crossover rate**, **mutation rate**, and **mutation strength** should be saved in the binary file header for future flexibility, and added the the CLI + TUI configuration screens / commands. This will enable easy comparison of different strategies.

2. **Deeper Analysis and Usability**
   - **TUI Benchmarking:** Create a TUI screen to **automatically benchmark** all engines for a set number of generations, displaying the results (e.g., time per generation, final fitness) in a clean table.
