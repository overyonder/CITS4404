use std::{
    thread,
    time::{Duration, Instant},
};

use macroquad::{miniquad::window::quit, prelude::*};
use pong_ai::{
    constants::*,
    drawing::*,
    game::{
        // controllers::Player,
        state::{FRAME_RATE, Game},
    },
    nn::Group,
};

#[macroquad::main("MyGame")]
async fn main() {
    let mut debugging: bool = true;
    // let left = &Player {
    //     up_key: KeyCode::W,
    //     down_key: KeyCode::S,
    // };
    // let right = &Player {
    //     up_key: KeyCode::Up,
    //     down_key: KeyCode::Down,
    // };

    let mut group = Group::new(POP_SIZE);
    group.inject_weights(&FORMER_CHAMPION, ELITES, POP_SIZE);
    let start = Instant::now();
    let mut round = Instant::now();
    for i in 0..GENERATIONS {
        let longest_match = group.train(TOURNAMENT_SIZE);
        group.individuals_mut().sort();
        if i % (GENERATIONS / 32) == 0 {
            println!("Generations: {}", GENERATIONS);
            println!("Generations complete: {}", (i + 1));
            println!("Population size: {}", POP_SIZE);
            println!("Elites: {}", ELITES);
            println!("Tournament size: {}", TOURNAMENT_SIZE);
            println!(
                "Matches done: {}",
                (i + 1) * POP_SIZE * (TOURNAMENT_SIZE - 1)
            );
            println!("Elapsed time: {:?}", start.elapsed());
            println!("Round time: {:?}", round.elapsed());
            println!(
                "Longest match: {} ticks of possible {}",
                longest_match,
                FRAME_RATE * 120.
            );
            println!("Matches per round: {}", POP_SIZE * (TOURNAMENT_SIZE - 1));
            round = Instant::now();
            println!(
                "Matches per second: {}",
                (POP_SIZE * (TOURNAMENT_SIZE - 1)) as f64 / round.elapsed().as_secs_f64()
            );
            println!(
                "Fitnesses: {:?}",
                group
                    .individuals()
                    .iter()
                    .map(|i| i.fitness())
                    .collect::<Vec<_>>()
            );
            println!(
                "Best weights: {:?}",
                group.individuals()[POP_SIZE - 1].weights()
            );
        }
        group.mutate(ELITES, POP_SIZE);
    }
    let left = &group.individuals()[POP_SIZE - 1];
    let right = &group.individuals()[POP_SIZE - 2];
    let mut game = Game::default();
    loop {
        // Top-level input events
        if is_key_released(KeyCode::Q) {
            quit();
        } else if is_key_released(KeyCode::N) {
            debugging = !debugging;
        }

        // Game loop
        clear_background(BLACK);
        draw_game(&debugging, screen_width(), screen_height(), &game, left, right);
        game.tick(left, right);

        // Loop end
        next_frame().await;
        thread::sleep(Duration::from_millis((1000. / FRAME_RATE) as u64));
    }
}
