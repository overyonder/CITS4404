mod constants;
mod gamestate;
mod individual;
mod population;
mod tui;

// IMPORTANT: Add `clap = { version = "4.0", features = ["derive"] }` to your [dependencies] in Cargo.toml
use clap::Parser;
use population::Population;
use std::io;
use tui::run_app;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The engine to use for the simulation
    #[arg(short, long)]
    engine: Option<String>,

    /// Number of generations to run the simulation for
    #[arg(short, long, default_value_t = 100)]
    generations: u32,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    #[cfg(debug_assertions)]
    println!("Debugging enabled. Args: {:?}", args);

    match args.engine.as_deref() {
        Some("stack") => {
            println!(
                "Running simulation on '{}' engine for {} generations.",
                "stack", args.generations
            );
            let mut population = Population::abiogenesis();
            population.evolve(args.generations);
        }
        Some("simd") => {
            println!("SIMD engine not yet implemented.");
        }
        Some("gpu") => {
            println!("GPU engine not yet implemented.");
        }
        Some(other) => {
            println!("Unknown engine type: {}", other);
        }
        None => {
            // No engine specified, run interactive TUI mode
            run_app(args.generations)?;
        }
    }

    Ok(())
}
