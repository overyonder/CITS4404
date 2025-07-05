mod config;
mod constants;
mod cpp_compat;
mod engines;
mod gamestate;
mod population;
mod traits;
mod tui;
mod utils;

use crate::{
    config::{Activation, Config, Engine, FitnessFunc},
    engines::{HeapIndividual, SimdIndividual, StackIndividual},
    population::Population,
    tui::ui::run_app,
};
use clap::Parser;
use std::io;

/// Command-line arguments for configuring the evolutionary algorithm and neural network engine.
///
/// This struct uses `clap` to define and parse command-line arguments, allowing for
/// flexible configuration of the application when running in headless/CLI mode.
///
/// # Teaching Note
///
/// This is a great example of using a declarative macro (`#[derive(Parser)]`) to automatically
/// generate a command-line parser. Each field corresponds to a potential command-line argument.
/// The `#[arg(...)]` attributes provide metadata like short/long names, help text, and default values.
/// This pattern separates configuration from application logic, making the code cleaner and easier to maintain.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The neural network engine to use for forward propagation.
    ///
    /// The choice of engine determines the underlying data structures and execution strategy for the
    /// neural network, with significant performance implications.
    ///
    /// - `stack`: Uses fixed-size arrays on the stack for genomes and populations. This is often the
    ///   fastest for smaller networks due to excellent cache locality and no dynamic allocation overhead.
    /// - `heap`: Uses dynamically sized `Vec<T>` for genomes and populations, providing flexibility
    ///   at the cost of potential cache misses and allocation overhead.
    /// - `simd`: Like `stack`, but leverages Single Instruction, Multiple Data (SIMD) instructions
    ///   to perform calculations on multiple data points in parallel, offering a significant speedup
    ///   on compatible CPUs.
    /// - `gpu`: Offloads the entire tournament simulation to the GPU using WGSL shaders, ideal for
    ///   massively parallel workloads.
    #[arg(short, long, default_value = "stack")]
    engine: String,

    /// Enables concurrent execution using the Rayon library.
    ///
    /// When this flag is present, the fitness evaluation of the population is parallelized
    /// across multiple CPU cores. This is available for `stack`, `heap`, and `simd` engines.
    /// It is a simple and effective way to leverage multi-core processors.
    #[arg(long)]
    concurrent: bool,

    /// The number of generations to run the evolutionary algorithm.
    ///
    /// A generation is a single cycle of evaluation, selection, crossover, and mutation.
    /// More generations increase the likelihood of finding a better solution but take longer to run.
    #[arg(short, long, default_value_t = 100)]
    generations: u32,

    /// The number of individuals (genomes) in the population.
    ///
    /// A larger population size increases genetic diversity, which can help avoid premature
    /// convergence to a local optimum. However, it also increases the computational cost of
    /// each generation.
    #[arg(long, default_value_t = 128)]
    population_size: usize,

    /// The number of the fittest individuals to carry over to the next generation unchanged.
    ///
    /// Elitism ensures that the best solutions found so far are not lost due to crossover or
    /// mutation. A higher elite count can speed up convergence but may also reduce diversity.
    #[arg(long, default_value_t = 2)]
    elite_count: usize,

    /// The probability (from 0.0 to 1.0) that a gene (weight) will be mutated.
    ///
    /// Mutation introduces new genetic material into the population, preventing stagnation.
    /// A higher mutation rate increases exploration of the search space but can disrupt
    /// good solutions if set too high.
    #[arg(long, default_value_t = 0.05)]
    mutation_rate: f32,

    /// The maximum magnitude of a mutation.
    ///
    /// When a gene is mutated, a random value between `-mutation_strength` and `+mutation_strength`
    /// is added to it. A larger strength allows for bigger jumps in the search space, which can
    /// be useful for escaping local optima.
    #[arg(long, default_value_t = 0.1)]
    mutation_strength: f32,

    /// The activation function to use in the neural network's hidden layers.
    ///
    /// The activation function introduces non-linearity, allowing the network to learn
    /// complex patterns. Different functions have different properties:
    /// - `tanh`: Hyperbolic tangent, squashes values to the range [-1, 1].
    /// - `relu`: Rectified Linear Unit, outputs `x` if `x > 0` and `0` otherwise. Computationally efficient.
    /// - `atan`: Arctangent, squashes values to the range [-PI/2, PI/2].
    /// - `linear`: No-op, simply passes the value through. Results in a linear model.
    #[arg(long)]
    activation: Option<String>,

    /// The fitness function to use for evolution.
    #[arg(long, value_enum, default_value_t = Config::default().fitness_func)]
    fitness_func: FitnessFunc,
}

/// Application entry point.
///
/// This function orchestrates the entire application flow. It parses command-line
/// arguments to determine whether to run in command-line interface (CLI) mode or
/// text-based user interface (TUI) mode.
///
/// # Modes of Operation
///
/// 1.  **TUI Mode (Default):** If the application is run without any command-line
///     arguments, it launches an interactive TUI. This mode is user-friendly and
///     allows for real-time control over training and simulation.
///
/// 2.  **CLI Mode:** If any command-line arguments are provided, the application
///     runs in a headless CLI mode. This is ideal for scripting, batch processing,
///     and performance benchmarking, as it runs the simulation with the specified
///     configuration and then exits.
///
/// # Teaching Note
///
/// The use of `std::env::args().len()` is a simple method to detect the presence of
/// CLI arguments. A more advanced approach might involve a dedicated `--tui` flag or
/// subcommand, but this direct approach is effective for this application's needs.
/// The logic cleanly separates the two main paths of the application, delegating
/// the complex setup and execution to dedicated functions (`run_cli` and `tui::ui::run_app`).
fn main() -> io::Result<()> {
    let args = Args::parse();

    // If arguments are provided (more than just the program name), run in CLI mode.
    // Otherwise, launch the interactive TUI.
    if std::env::args().len() > 1 {
        run_cli(args);
    } else {
        if let Err(e) = run_app() {
            // Use eprintln to write to standard error, a common practice for errors.
            eprintln!("TUI Application Error: {}", e);
        }
    }

    Ok(())
}

/// Runs the application in headless (CLI) mode with the given configuration.
///
/// This function takes the parsed command-line arguments, constructs the
/// `EvolutionConfig`, and runs the evolutionary algorithm for the specified
/// number of generations using the chosen engine.
///
/// # Algorithm Steps
///
/// 1.  **Engine Selection:** Determines the neural network engine (`Stack`, `Heap`, etc.)
///     and whether to use concurrent execution based on the `engine` and `concurrent` arguments.
/// 2.  **Configuration Building:** Creates an `EvolutionConfig` struct. It starts with
///     default values and overrides them with any values provided via CLI arguments.
///     This makes the CLI flexible, as users only need to specify what they want to change.
/// 3.  **Population Initialization:** A `Population` of the appropriate `Individual`
///     type (e.g., `StackIndividual`, `SimdIndividual`) is created based on the config.
/// 4.  **Evolution Loop:** The `population.evolve()` method is called, which runs the
///     main genetic algorithm loop (evaluation, selection, crossover, mutation) for the
///     configured number of generations.
/// 5.  **Output:** Prints the configuration and progress to the console.
///
/// # Teaching Note
///
/// The use of a generic `Population<T>` struct, where `T` implements the `Individual`
/// trait, is a powerful Rust pattern. It allows the same core evolutionary logic to
/// operate on different underlying data representations (the engines) without code
/// duplication. The `match` statement on `config.engine` is the dispatch point that
/// selects the concrete type for the generic parameter `T` at compile time.
fn run_cli(args: Args) {

    // The `concurrent` flag is now just a boolean in the config, not part of the engine enum.
    // We parse the engine string directly.
    let engine = match args.engine.as_str() {
        "stack" => Engine::Stack,
        "simd" => Engine::Simd,
        "heap" => Engine::Heap,
        "gpu" => Engine::Gpu,
        other => {
            eprintln!("Unknown engine: {}. Exiting.", other);
            return;
        }
    };

    // Build config from CLI arguments. Since we now have default values in `Args`,
    // we can construct the config directly without checking `Option`s.
    let mut config = Config {
        generations: args.generations,
        engine,
        concurrent: args.concurrent,
        population_size: args.population_size,
        elite_count: args.elite_count,
        mutation_rate: args.mutation_rate,
        mutation_strength: args.mutation_strength,
        fitness_func: args.fitness_func,
        ..Default::default() // Use default for activation, which is handled separately
    };
    if let Some(ref act) = args.activation {
        config.activation = match act.to_lowercase().as_str() {
            "tanh" => Activation::Tanh,
            "relu" => Activation::Relu,
            "atan" => Activation::Atan,
            "linear" => Activation::Linear,
            _ => {
                println!("Unknown activation: {}. Using default.", act);
                config.activation
            }
        };
    }

    // The `Display` trait is used for printing, which is more idiomatic than a custom method.
    println!("Running in headless mode with engine: {}", config.engine);
    println!("Configuration: {:#?}", config);

    let evolution_callback = |gen, best_fitness, avg_fitness, worst_fitness, _genome: &[f32]| {
        println!(
            "Gen {:<4}/ {} | Best: {:<5} | Avg: {:<7.2} | Worst: {}",
            gen, config.generations, best_fitness, avg_fitness, worst_fitness
        );
        true // Return true to continue the evolution.
    };

    println!("Starting evolution...");
    match config.engine {
        Engine::Stack => {
            let mut pop: Population<StackIndividual> = Population::new(config);
            pop.evolve(evolution_callback);
        }
        Engine::Heap => {
            let mut pop: Population<HeapIndividual> = Population::new(config);
            pop.evolve(evolution_callback);
        }
        Engine::Simd => {
            let mut pop: Population<SimdIndividual> = Population::new(config);
            pop.evolve(evolution_callback);
        }
        Engine::Gpu => {
            println!("GPU engine is not yet supported in CLI mode.");
        }
    }
    println!("Evolution finished.");
}
