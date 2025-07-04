use std::io::{self, Write};

// Project modules
mod constants;
mod evolution;
mod game;
mod net;
mod paddle;
mod tui;

const BEST_NET_FILE: &str = "best_net.json";

fn main() -> io::Result<()> {
    loop {
        println!("\n--- Pong AI Menu ---");
        println!("1. Train new model");
        println!("2. Run simulation with best model");
        println!("3. Exit");
        print!("Please enter your choice: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;

        match choice.trim() {
            "1" => {
                println!("\nHow many generations should the training run for? (e.g., 100)");
                print!("> ");
                io::stdout().flush()?;
                let mut generations_str = String::new();
                io::stdin().read_line(&mut generations_str)?;
                let generations = generations_str.trim().parse().unwrap_or(100);
                evolution::run_evolution(generations)?;
            }
            "2" => {
                println!("\nStarting simulation...");
                // Load the best network and create paddles from it.
                let left_paddle = paddle::Paddle::from_best_net(BEST_NET_FILE);
                let right_paddle = paddle::Paddle::from_best_net(BEST_NET_FILE);

                // Create a new game state with the trained paddles.
                let game_state = game::GameState::new_with_paddles(left_paddle, right_paddle);

                // Render the game with the loaded state.
                tui::render_game(Some(game_state))?;
            }
            "3" => {
                println!("Exiting.");
                break;
            }
            _ => {
                println!("Invalid choice, please try again.");
            }
        }
    }

    Ok(())
}
