mod config;
mod constants;
mod cpp_compat;
mod engines;
mod gamestate;
mod population;
mod traits;
mod tui;
mod utils;

use std::{fs, io, path::Path};

use clap::Parser;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use crate::{
    config::{Config, Engine},
    engines::StackIndividual,
    gamestate::GameState,
    population::Population,
    traits::Individual,
    tui::ui::run_app,
};

#[derive(Parser)]
#[command(name = "CITS4404-Project")]
#[command(about = "A neural network training application using evolutionary algorithms")]
#[command(version)]
struct Args {
    /// The neural network engine to use for forward propagation.
    #[arg(short, long, default_value = "stack")]
    engine: String,

    /// Enables concurrent execution using the Rayon library.
    #[arg(long)]
    concurrent: bool,

    /// The number of generations to run the evolutionary algorithm.
    #[arg(short, long, default_value_t = 100)]
    generations: u32,

    /// The number of individuals (genomes) in the population.
    #[arg(long, default_value_t = 128)]
    population_size: usize,

    /// The number of the fittest individuals to carry over to the next generation unchanged.
    #[arg(long, default_value_t = 2)]
    elite_count: usize,

    /// The probability (from 0.0 to 1.0) that a gene (weight) will be mutated.
    #[arg(long, default_value_t = 0.05)]
    mutation_rate: f32,

    /// The maximum magnitude of a mutation.
    #[arg(long, default_value_t = 0.1)]
    mutation_strength: f32,

    /// The activation function to use in the neural network's hidden layers.
    #[arg(long)]
    activation: Option<String>,

    /// The mutation strategy to use for evolution.
    #[arg(long, value_enum, default_value_t = Config::default().mutation_strategy)]
    mutation_strategy: crate::config::MutationStrategy,

    /// The fitness function to use for evolution.
    #[arg(long, value_enum, default_value_t = Config::default().fitness_func)]
    fitness_func: crate::config::FitnessFunc,

    /// Path to save the best individual to after training is complete.
    #[arg(long)]
    save_to: Option<String>,

    /// Path to load an individual from and run a simulation.
    /// When using this, other training arguments are ignored.
    #[arg(long)]
    load_from: Option<String>,
}

/// Application entry point.
fn main() -> io::Result<()> {
    // === Enhanced Logging Configuration ===
    // Configure tracing subscriber with conditional console output
    // In TUI mode, we disable console logging to avoid disrupting the interface
    // In CLI mode, we enable full logging for debugging and monitoring
    let args = Args::parse();
    if args.load_from.is_some() || std::env::args().len() > 1 {
        // CLI mode: Enable full console logging
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env().add_directive("rust=info".parse().unwrap()))
            .init();
    } else {
        // TUI mode: Suppress console output to prevent UI disruption
        // Only log to stderr at error level to avoid interfering with the TUI
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_env_filter(EnvFilter::from_default_env().add_directive("rust=error".parse().unwrap()))
            .init();
    }

    // This allows `tracing` macros (info!, error!, etc.) to be captured
    // by the subscriber we just configured above.

    let args = Args::parse();

    // Dispatch to the correct function based on arguments.
    // `load_from` takes precedence and runs a simulation.
    // Otherwise, if other args are present, run the CLI trainer.
    // If no args are present, run the TUI.
    if let Some(path) = &args.load_from {
        if let Err(e) = run_simulation_from_file(path) {
            error!("Error running simulation from file: {}", e);
        }
    } else if std::env::args().len() > 1 {
        run_cli(args);
    } else {
        if let Err(e) = run_app() {
            error!("TUI Application Error: {}", e);
        }
    }

    Ok(())
}

/// Runs the application in headless (CLI) mode with the given configuration.
fn run_cli(args: Args) {
    println!("--- RUNNING IN CLI MODE ---"); // Debug print
    let engine = match args.engine.as_str() {
        "cpu" => Engine::Cpu,
        "gpu" => Engine::Gpu,
        #[cfg(feature = "torch")]
        "torch" | "pytorch" => Engine::Torch,
        other => {
            error!("Unknown engine: {}. Available options: cpu, gpu{}. Exiting.", other, 
                   if cfg!(feature = "torch") { ", torch" } else { "" });
            return;
        }
    };

    let mut config = Config {
        generations: args.generations,
        engine,
        concurrent: args.concurrent,
        population_size: args.population_size,
        elite_count: args.elite_count,
        mutation_rate: args.mutation_rate,
        mutation_strength: args.mutation_strength,
        mutation_strategy: args.mutation_strategy,
        fitness_func: args.fitness_func,
        ..Default::default()
    };

    if let Some(ref act) = args.activation {
        config.activation = match act.to_lowercase().as_str() {
            "clampedlinear" | "clamped_linear" | "clamped-linear" => crate::config::Activation::ClampedLinear,
            "tanh" => crate::config::Activation::Tanh,
            "relu" => crate::config::Activation::Relu,
            "atan" => crate::config::Activation::Atan,
            "sigmoid" => crate::config::Activation::Sigmoid,
            "linear" => crate::config::Activation::Linear,
            _ => {
                warn!("Unknown activation: {}. Using default.", act);
                config.activation
            }
        };
    }

    info!("Running in headless mode with engine: {}", config.engine);
    info!("Starting evolution...");

    // This macro reduces code duplication for running evolution and saving the best individual.
    // It uses static dispatch by creating a new population of a specific Individual type.
    macro_rules! run_evolution {
        ($individual_type:ty, $config:expr, $args:expr) => {{
            let mut pop: Population<$individual_type> = Population::new($config.clone());
            // In CLI mode, we don't have a UI, so we pass `None` for the sender.
            // The `evolve` function will print progress to the console when sender is `None`.
            let best_individual = pop.evolve(None);
            if let Some(filename) = &$args.save_to {
                let save_path = Path::new("models").join(filename);
                // Create the directory if it doesn't exist.
                if let Some(parent_dir) = save_path.parent() {
                    if let Err(e) = fs::create_dir_all(parent_dir) {
                        error!("Failed to create directory {}: {}", parent_dir.display(), e);
                        return;
                    }
                }

                if let Some(path_str) = save_path.to_str() {
                    if let Err(e) = best_individual.save(path_str, &$config) {
                        error!("Failed to save best individual: {}", e);
                    } else {
                        info!("Best individual saved to {}", path_str);
                    }
                } else {
                    error!("Invalid save path created.");
                }
            }
        }};
    }

    // We use the macro to generate type-specific code for each engine, avoiding dyn Trait.
    match config.engine {
        Engine::Cpu => run_evolution!(StackIndividual, config, args),
        // GPU batch processing uses lightweight StackIndividual with GpuBatchEngine for GPU operations
        Engine::Gpu => run_evolution!(StackIndividual, config, args),
        #[cfg(feature = "torch")]
        Engine::Torch => {
            #[cfg(feature = "torch")]
            {
                use crate::engines::TorchIndividual;
                run_evolution!(TorchIndividual, config, args)
            }
        }
    };

    info!("Evolution finished.");
}

/// Loads an individual from a file and runs a single game simulation in the console.
fn run_simulation_from_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Loading model from: {}", path);
    let (weights, config) = tui::model_loader::load_model_from_file(Path::new(path))?;

    info!("--- Starting Simulation ---");
    info!(" Model: {}", config.name.as_deref().unwrap_or("N/A"));
    info!(" Engine: {}", config.engine);
    info!(" Generations: {}", config.generations);
    info!(" Activation: {}", config.activation);
    info!("---------------------------");

    let individual = StackIndividual { 
        weights: weights.try_into().expect("Invalid weights array size")
    };
    run_game_simulation(&individual, &config);

    Ok(())
}

/// Runs a single game simulation between two instances of the same individual.
fn run_game_simulation<I: Individual>(individual: &I, config: &Config) {
    let mut game_state = GameState::new();
    let ((left_primary, _), (right_primary, _)) =
        game_state.simulate(individual, individual, config);

    info!("--- Simulation Finished ---");
    info!("Final Score: {} - {}", game_state.scores.0, game_state.scores.1);
    info!(
        "Left Player Fitness (Returns+Shots): {}",
        left_primary
    );
    info!(
        "Right Player Fitness (Returns+Shots): {}",
        right_primary
    );
}
