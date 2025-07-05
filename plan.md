# Cascade's Core Principles & Project Plan

This document outlines the core principles for our development workflow and the master plan for the Neural Pong project. These guidelines are based on lessons learned from our recent collaboration and are designed to ensure accuracy, efficiency, and alignment with your goals.

---

## Part 1: Core Principles for Effective and Error-Free Development

### Tool Usage Best Practices

1.  **Principle: Always Read Before Writing.**

    - **Action:** Before any file modification (`replace_file_content`), I will read the _entire_ file content or the specific, relevant function using `view_line_range` or `view_code_item`, noting that by default it only reads the first 400 lines unless I specify otherwise.
    - **Reasoning:** Relying on outlines or partial reads has repeatedly caused my edits to fail because the `TargetContent` was not exact. This led to mangled code and a cascade of syntax errors, as we saw in [tui/ui.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/tui/ui.rs:0:0-0:0). A full read ensures my understanding of the code is complete and my edits are precise.

2.  **Principle: Verify Context Before Changing APIs or Logic.**

    - **Action:** Before altering any public-facing code (like a struct name or function signature), I will use `grep_search` or `codebase_search` to find all its usages across the entire project.
    - **Reasoning:** Making changes in isolation creates compilation errors in other files. A codebase-wide search reveals the full impact of a change, allowing me to perform a comprehensive, multi-file fix in a single, logical step, preventing unresolved import errors and other downstream issues.

3.  **Principle: Trust, but Verify, Linter Feedback.**
    - **Action:** I will treat the latest list of lint/compiler errors you provide as the absolute source of truth for the current state of the codebase.
    - **Reasoning:** My internal state can become stale. Ignoring fresh linter feedback led to a frustrating loop of fixing problems that had already been fixed or were no longer relevant. The compiler does not lie.

### Preferred Workflow Methodology

1.  **Step 1: Plan and Prioritize.**

    - **Action:** I will begin each major task by updating the master plan. I will break down your requests into a clear, ordered checklist.
    - **Reasoning:** A clear plan prevents reactive, ad-hoc changes. It ensures we tackle root causes first (e.g., fixing [config.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/config.rs:0:0-0:0) before the TUI files that depend on it) and provides a shared understanding of the path forward.

2.  **Step 2: Investigate Systematically.**

    - **Action:** I will execute the principles of "Read Before Writing" and "Verify Context." I will gather all necessary information _before_ formulating a solution. I will not guess or assume.
    - **Reasoning:** Hasty implementation without full context was the primary driver of our recent failures. A dedicated investigation phase prevents wasted effort and incorrect edits.

3.  **Step 3: Execute with Precision.**

    - **Action:** I will apply changes in a single, logical, and comprehensive tool call whenever possible. For multi-file changes, I will group them into one response.
    - **Reasoning:** Atomic changes are easier to track and debug. This avoids leaving the codebase in a broken intermediate state.

4.  **Step 4: Verify and Iterate.**
    - **Action:** After applying a change, I will immediately request a build or check command (`cargo check`) to confirm the fix and get immediate feedback. I will use the new list of errors to guide my next step.
    - **Reasoning:** This creates a tight feedback loop, ensuring each step successfully builds upon the last and preventing the accumulation of errors.

---

## Part 2: Current Master Plan

### Neural Network Pong Application Review Plan

#### Notes

- The application is a neural network-based Pong game with both CLI and TUI modes.
- TUI mode should allow users to train, visualize, and simulate neural nets—these features must be checked for completeness.
- The genetic algorithm optimizes neural network parameters for paddle control; multiple engine types are supported (Stack, Heap, SIMD, Concurrent, GPU).
- Fitness is based on paddle-ball interactions; all game logic must follow described physical rules.
- The fitness function should be modular and user-selectable (C++-equivalent, current, and a third improved version) in both CLI and TUI.
- Rustdoc comments must thoroughly document algorithms, data structures, and memory allocation, as this code is for teaching purposes.
- README.md should be updated to explain the application, algorithms, settings, and provide a comparison to the legacy C++ version.
- A compatibility layer is required to use C++-generated weights in the Rust simulation.
- Benchmarking scripts should compare all engine types and the C++ version using hyperfine.
- Do not make code changes until all files have been reviewed.
- Core Rust files ([main.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/main.rs:0:0-0:0), [config.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/config.rs:0:0-0:0), `constants.rs`, [gamestate.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/gamestate.rs:0:0-0:0), [population.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/population.rs:0:0-0:0), [traits.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/traits.rs:0:0-0:0)) have been reviewed and are well-structured/documented.
- Engine implementation review (stack, heap, simd, gpu) is complete.
- Noted code duplication in helper functions ([dot](cci:1://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/simd.rs:69:0-73:1), `apply_activation`) across stack, heap, and simd engines; consider refactoring for reuse.
- `apply_activation` does not handle all Activation enum variants (e.g., Atan, Linear) in all engines; this needs fixing.
- In [gpu.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/gpu.rs:0:0-0:0), [weights_as_slice](cci:1://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/gpu.rs:305:4-307:5) and [weights_as_mut_slice](cci:1://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/simd.rs:138:4-140:5) are unimplemented, violating trait contract.
- In [gpu.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/gpu.rs:0:0-0:0), activation mapping mismatches between host and shader; Atan/Linear missing, Sigmoid present but not in enum.
- GPU engine has performance bottlenecks: crossover/mutate require full GPU-CPU transfers; future optimization should move these to compute shaders.
- The forward.wgsl shader has been reviewed and matches the expected architecture, but activation handling and host/shader enum sync require attention.
- TUI core files (`mod.rs`, [app.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/tui/app.rs:0:0-0:0), [ui.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/tui/ui.rs:0:0-0:0), [training.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/tui/training.rs:0:0-0:0), `simulation.rs`) have been reviewed; structure is modular and features are mostly present.
- GPU engine is not supported in the TUI (training.rs); this is a missing feature to address.
- Benchmarking script (benchmark.sh) has been reviewed; it benchmarks Rust and C++ engines, but has issues: inconsistent C++ binary path, missing concurrent Rust engine, and hardcoded parameters.
- User requests major CLI and TUI UX improvements:
  - CLI: Print progress (generation, best fitness) to stdout each generation.
  - CLI: All parameters should show default values in --help.
  - TUI: Add configuration screen before training; do not use only defaults.
  - TUI: Error messages must persist until dismissed by user. (Now implemented and tested)
  - TUI: Training dashboard should be feature-rich (progress bar, engine badge, sparklines, stopwatch, bar chart for genome, etc.)—reference ratatui demo for inspiration.
- Fitness function enum has been added to config; CLI/TUI integration for fitness selection is complete.
- CLI and TUI now allow user selection of fitness function; integration complete.

#### Task List

- [x] Review all Rust source files for completeness and correctness (main.rs, config.rs, constants.rs, gamestate.rs, population.rs, traits.rs)
  - [x] Review all engine implementations (stack, heap, simd, gpu)
  - [x] Review forward.wgsl shader for GPU engine
  - [x] Review TUI core files (mod.rs, app.rs, ui.rs, training.rs, simulation.rs)
  - [x] Review benchmarking script (benchmark.sh)
- [x] Identify missing or incomplete features (especially TUI training/simulation/visualization)
- [x] Check all game logic for compliance with described physical rules
- [x] Ensure all engine types are benchmarkable
- [x] Confirm thorough rustdoc comments (algorithms, data structures, memory allocation)
- [x] Update README.md with structured documentation and comparison to C++
- [x] Plan compatibility layer for C++ weights in Rust
- [x] List and analyze current problems (@current_problems)
- [x] Refactor helper functions ([dot](cci:1://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/simd.rs:69:0-73:1), `apply_activation`) for reuse
- [x] Ensure all Activation enum variants are handled in all engines
- [x] Implement [weights_as_slice](cci:1://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/gpu.rs:305:4-307:5) and [weights_as_mut_slice](cci:1://file:///c:/Users/user/_dev/CITS4404/rust/src/engines/simd.rs:138:4-140:5) in gpu.rs
- [x] Sync activation mapping between host and shader in GPU engine
- [x] Add GPU engine support to TUI training mode
- [x] Only after review: Plan and execute necessary code/documentation changes
- [x] Make TUI error messages persistent until dismissed
- [x] Implement CLI progress reporting (stdout per generation)
- [x] Ensure CLI --help prints all parameters with defaults
- [x] Add TUI configuration screen before training
- [x] Redesign TUI training dashboard with advanced features (progress bar, engine badge, sparklines, stopwatch, genome chart, etc.)
- [ ] After above: Test C++ compatibility layer and optimize GPU engine
- [ ] Refactor fitness function to be modular/selectable (C++-equivalent, current, improved)
  - [x] Implement C++-equivalent fitness function enum and config integration
  - [ ] Implement C++-equivalent fitness function logic
  - [ ] Implement current Rust fitness function as a selectable option
  - [ ] Design and implement a third, improved fitness function (e.g., address "quick win" reward issue)
  - [x] Add CLI/TUI option to select fitness function
  - [ ] Allow parameterization of fitness function (weights for returns, time, shots, wins, etc.)
  - [ ] Update documentation and help text for fitness selection
- [ ] Write idiomatic rust tests for each major feature of the application to enable early identification of breaking changes as new features are added in the future

#### Current Goal

- Fix all remaining compilation errors in [tui/ui.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/tui/ui.rs:0:0-0:0).
