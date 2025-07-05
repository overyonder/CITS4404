mod config;
mod constants;
mod engines;
mod gamestate;
mod population;
mod traits;
mod tui;

use crate::{
    config::{Engine, EvolutionConfig},
    engines::{GpuIndividual, HeapIndividual, SimdIndividual, StackIndividual},
    population::Population,
    tui::run_app,
};
use clap::Parser;
use std::io;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The engine to use for the simulation
    #[arg(short, long)]
    engine: Option<String>,

    #[arg(long)]
    concurrent: bool,

    /// Number of generations to run the simulation for
    #[arg(short, long, default_value_t = 100)]
    generations: u32,

    /// Run without the TUI
    #[arg(long)]
    nogui: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if args.nogui {
        let engine_str = args.engine.as_deref().unwrap_or("stack");
        let concurrent = args.concurrent;
        let engine = match (engine_str, concurrent) {
            ("stack", false) => Engine::Stack,
            ("stack", true) => Engine::ConcurrentStack,
            ("simd", false) => Engine::Simd,
            ("simd", true) => Engine::ConcurrentSimd,
            ("gpu", _) => Engine::Gpu,
            ("heap", _) => Engine::Heap,
            (other, _) => {
                println!("Unknown engine: {}", other);
                return Ok(());
            }
        };

        // In no-gui mode, we still use the config struct, but we populate it from args.
        // We can use default values for parameters not exposed via CLI args.
        let config = EvolutionConfig {
            generations: args.generations,
            engine,
            concurrent,
            ..Default::default()
        };

        println!(
            "Running in headless mode with config: generations={}, engine={}",
            config.generations,
            config.engine.to_str()
        );

        println!("Running in headless mode with config: {:#?}", config);

        match config.engine {
            Engine::Stack | Engine::ConcurrentStack => {
                let mut population: Population<StackIndividual> = Population::new(config);
                population.evolve();
            }
            Engine::Simd | Engine::ConcurrentSimd => {
                let mut population: Population<SimdIndividual> = Population::new(config);
                population.evolve();
            }
            Engine::Heap | Engine::ConcurrentHeap => {
                let mut population: Population<HeapIndividual> = Population::new(config);
                population.evolve();
            }
            Engine::Gpu => {
                let mut population: Population<GpuIndividual> = Population::new(config);
                population.evolve();
            }
        }
    } else {
        // Run with the interactive TUI.
        // The TUI now manages its own config state.
        if let Err(e) = run_app() {
            println!("Application error: {}", e);
        }
    }

    Ok(())
}
