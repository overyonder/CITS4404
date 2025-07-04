mod constants;
mod evolution;
mod game;
mod menu;
mod net;
mod paddle;
mod tui;

use clap::{Parser, ValueEnum};
use evolution::run_evolution_stack;
use menu::run_app;
use std::io;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The computation engine to use for training.
    #[arg(short, long)]
    engine: Option<Engine>,

    /// The number of generations to run training for.
    #[arg(short, long, default_value_t = 100)]
    generations: u32,
}

#[derive(ValueEnum, Clone, Debug)]
enum Engine {
    Stack,
    Simd,
    Gpu,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if let Some(engine) = args.engine {
        // Benchmark mode: run directly from command line arguments
        println!(
            "Running in benchmark mode with engine: {:?}, for {} generations",
            engine, args.generations
        );
        match engine {
            Engine::Stack => run_evolution_stack(args.generations)?,
            Engine::Simd => {
                println!("SIMD engine not yet implemented.");
            }
            Engine::Gpu => {
                println!("GPU engine not yet implemented.");
            }
        }
    } else {
        // Interactive mode: run the Ratatui menu app
        run_app()?;
    }

    Ok(())
}
