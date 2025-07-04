use std::io;
use std::env;
use crossterm::{terminal::{enable_raw_mode, EnterAlternateScreen}, execute};
use ratatui::{prelude::*, widgets::*};

// Constants module
mod constants;
mod game;
mod net;
mod paddle;

use crate::{
    constants::{ELITISM_RATIO, MAX_SCORE, POPULATION_SIZE},
    game::GameState,
    net::Net,
    paddle::Paddle,
};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let generations = if args.len() > 1 {
        args[1].parse().unwrap_or(100)
    } else {
        100
    };
    println!("Running for {} generations...", generations);

    // 1. Initialize a population of neural networks.
    let mut population: Vec<Net> = (0..POPULATION_SIZE).map(|_| Net::new()).collect();

    // 2. Loop for the specified number of generations.
    for gen in 0..generations {
        // a. Evaluate fitness using a round-robin tournament where each network plays every other network.
        let mut fitness_scores = vec![0.0; POPULATION_SIZE];
        for i in 0..POPULATION_SIZE {
            for j in (i + 1)..POPULATION_SIZE {
                let (fitness1, fitness2) = run_training_game(&population[i], &population[j]);
                fitness_scores[i] += fitness1;
                fitness_scores[j] += fitness2;
            }
        }

        // b. Combine networks and their fitness scores for sorting.
        let mut ranked_population: Vec<(Net, f64)> = population
            .into_iter()
            .zip(fitness_scores.into_iter())
            .collect();

        // c. Sort the population by fitness in descending order.
        ranked_population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!(
            "Generation {}: Best Fitness = {:.2}",
            gen, ranked_population[0].1
        );

        // d. Create the next generation.
        population = create_next_generation(&ranked_population);

        // e. Optionally, render a game with the best network.
        // if gen % 10 == 0 {
        //     println!("Rendering best of generation {}", gen);
        //     render_game(&ranked_population[0].0)?;
        // }
    }

    println!("Training complete.");
    Ok(())
}

/// Creates a new generation of networks from the ranked parents of the previous generation.
fn create_next_generation(ranked_population: &[(Net, f64)]) -> Vec<Net> {
    let mut next_generation = Vec::with_capacity(POPULATION_SIZE);

    // 1. Elitism: Keep the best-performing networks.
    let elite_count = (ELITISM_RATIO * POPULATION_SIZE as f64).round() as usize;
    for i in 0..elite_count {
        // Add the elite networks directly to the next generation.
        next_generation.push(ranked_population[i].0.clone());
    }

    // 2. Crossover & Mutation: Fill the rest of the new generation.
    let remaining = POPULATION_SIZE - elite_count;
    for i in 0..remaining {
        // Select a random elite parent to be the basis for a new offspring.
        let parent = &ranked_population[i % elite_count].0;
        let mut offspring = parent.clone();
        // Mutate the offspring to introduce variation.
        offspring.mutate();
        next_generation.push(offspring);
    }

    next_generation
}

/// Runs a game simulation between two neural networks and returns their fitness scores.
fn run_training_game(net1: &Net, net2: &Net) -> (f64, f64) {
    let mut game = GameState::new();
    game.paddles[0].net = net1.clone();
    game.paddles[1].net = net2.clone();

    // Simulate the game until a player reaches the max score.
    while game.score[0] < MAX_SCORE && game.score[1] < MAX_SCORE {
        game.tick();
        // Add a safeguard to prevent infinite loops in case of a stalemate.
        if game.ticks > 60 * 1000 { // 1000 seconds @ 60 TPS
            break;
        }
    }

    // Fitness is calculated based on how long the game lasted (ticks),
    // how many times the ball was returned, and a penalty for shots taken.
    // This formula is based on the C++ implementation.
    let fitness1 = (game.ticks as f64) * 1000.0 + (game.returns[0] as f64) * 100.0
        - (game.shots[0] as f64).powf(2.0);
    let fitness2 = (game.ticks as f64) * 1000.0 + (game.returns[1] as f64) * 100.0
        - (game.shots[1] as f64).powf(2.0);

    (fitness1, fitness2)
}

/// Renders a full game in the terminal using the TUI.
fn render_game(_net: &Net) -> io::Result<()> {
    // Setup the terminal
    let mut terminal = setup_terminal()?;

    // Create game state
    let mut game_state = GameState::new();

    // Main game loop
    loop {
        terminal.draw(|f| ui(f, &game_state))?;

        game_state.tick();

        // TODO: Add logic to handle input and exit conditions
        // For now, we can just run for a fixed number of ticks.
        if game_state.ticks > 1000 {
            break;
        }
    }

    // Restore the terminal
    restore_terminal()
}

/// Sets up the terminal for TUI rendering.
fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

/// Restores the terminal to its original state.
fn restore_terminal() -> io::Result<()> {
    // Implementation details would go here...
    Ok(())
}

/// Defines the UI layout and widgets.
fn ui(frame: &mut Frame, game_state: &GameState) {
    // This is where you define what to draw on the screen.
    // 1. Create a main layout (e.g., a central block for the play area).
    // 2. Draw the play area boundaries.
    // 3. Draw the ball at its current position.
    // 4. Draw the left and right paddles at their positions.
    // 5. Draw the scores.
    let main_block = Block::default().borders(Borders::ALL).title("Pong");
    frame.render_widget(main_block, frame.size());
    // More rendering logic would go here...
}

