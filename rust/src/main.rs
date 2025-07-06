mod config;
mod constants;
mod cpp_compat;
mod engines;
mod gamestate;
mod population;
mod traits;
mod tui;
mod utils;

use crate::traits::Individual;
use crate::{
    config::{Activation, Config, Engine, FitnessFunc},
    engines::{GpuIndividual, HeapIndividual, SimdIndividual, StackIndividual},
    population::Population,
    tui::ui::run_app,
};
use clap::Parser;
use std::io::{self, Read};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

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

    /// The fitness function to use for evolution.
    #[arg(long, value_enum, default_value_t = Config::default().fitness_func)]
    fitness_func: FitnessFunc,

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
    // Initialize the logging subscriber.
    // You can override the log level by setting the RUST_LOG environment variable.
    // For example: `RUST_LOG=debug` or `RUST_LOG=trace`
    tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

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
    let engine = match args.engine.as_str() {
        "stack" => Engine::Stack,
        "simd" => Engine::Simd,
        "heap" => Engine::Heap,
        "gpu" => Engine::Gpu,
        other => {
            error!("Unknown engine: {}. Exiting.", other);
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
        fitness_func: args.fitness_func,
        ..Default::default()
    };

    if let Some(ref act) = args.activation {
        config.activation = match act.to_lowercase().as_str() {
            "tanh" => Activation::Tanh,
            "relu" => Activation::Relu,
            "atan" => Activation::Atan,
            "linear" => Activation::Linear,
            _ => {
                warn!("Unknown activation: {}. Using default.", act);
                config.activation
            }
        };
    }

    info!("Running in headless mode with engine: {}", config.engine);
    info!("Configuration: {:#?}", &config);

    let evolution_callback = |gen, best_fitness, avg_fitness, worst_fitness, _genome: &[f32]| {
        info!(
            "Gen {:<4}/ {} | Best: {:<5} | Avg: {:<7.2} | Worst: {}",
            gen, config.generations, best_fitness, avg_fitness, worst_fitness
        );
        true
    };

    info!("Starting evolution...");

    // This macro reduces code duplication for running evolution and saving the best individual.
    // It uses static dispatch by creating a new population of a specific Individual type.
    macro_rules! run_evolution {
        ($individual_type:ty, $config:expr, $args:expr, $callback:expr) => {{
            let mut pop: Population<$individual_type> = Population::new($config.clone());
            let best_individual = pop.evolve($callback);
            if let Some(path) = &$args.save_to {
                if let Err(e) = best_individual.save(path, &$config) {
                    error!("Failed to save best individual: {}", e);
                } else {
                    info!("Best individual saved to {}", path);
                }
            }
        }};
    }

    // We use the macro to generate type-specific code for each engine, avoiding dyn Trait.
    match config.engine {
        Engine::Stack => {
            run_evolution!(StackIndividual, config, args, evolution_callback)
        }
        Engine::Heap => {
            run_evolution!(HeapIndividual, config, args, evolution_callback)
        }
        Engine::Simd => {
            run_evolution!(SimdIndividual, config, args, evolution_callback)
        }
        Engine::Gpu => {
            run_evolution!(GpuIndividual, config, args, evolution_callback)
        }
    };

    info!("Evolution finished.");
}

/// Loads an individual from a file and runs a single game simulation.
fn run_simulation_from_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;

    // Read the configuration metadata first to determine the engine.
    let mut config_len_bytes = [0u8; 8];
    file.read_exact(&mut config_len_bytes)?;
    let config_len = u64::from_le_bytes(config_len_bytes);

    let mut config_bytes = vec![0u8; config_len as usize];
    file.read_exact(&mut config_bytes)?;
    let config: Config = serde_json::from_slice(&config_bytes)?;

    info!("Loaded model trained with engine: {}", config.engine);
    info!("Configuration: {:#?}", &config);

    // This macro reduces duplication for loading and running a simulation.
    // It ensures the correct, statically-typed `load` function is called.
    macro_rules! load_and_simulate {
        ($individual_type:ty, $path:expr) => {{
            // The file path is passed to `load`, which will re-open and read it.
            // This is simpler than trying to manage the file handle across functions.
            let (individual, config) = <$individual_type>::load($path)?;
            run_game_simulation(&individual, &config);
        }};
    }

    // Dispatch to the correct loader based on the engine specified in the loaded config.
    match config.engine {
        Engine::Stack => load_and_simulate!(StackIndividual, path),
        Engine::Heap => load_and_simulate!(HeapIndividual, path),
        Engine::Simd => load_and_simulate!(SimdIndividual, path),
        Engine::Gpu => load_and_simulate!(GpuIndividual, path),
    }

    Ok(())
}

/// Runs a single game between a loaded individual and a default opponent of the same type.
fn run_game_simulation<I: Individual>(individual: &I, config: &Config) {
    info!("\nRunning simulation...");
    // Create a default opponent of the *same type* as the loaded individual.
    // This is required because the `simulate` function expects both individuals to be the same type.
    let opponent = I::default();
    let mut game_state = gamestate::GameState::new();

    // The simulate function runs a game until MAX_SCORE is reached and returns the fitness scores.
    game_state.simulate(individual, &opponent, config);
    let (p1_score, p2_score) = game_state.scores;

    info!("Simulation Finished!");
    info!(
        "Final Score: Trained Model {} - {} Random Opponent",
        p1_score, p2_score
    );
}
