use macroquad::{
    miniquad::window::{self, quit},
    prelude::*,
};
use pong_ai::{
    constants::*,
    drawing::*,
    game::{
        controllers::Player,
        state::{FRAME_RATE, Game, Side},
    },
    nn::{Group, Individual},
};

#[macroquad::main("MyGame")]
async fn main() {
    draw_text("Loading...", 0., FONT_SIZE * 0.5, FONT_SIZE, YELLOW);
    #[cfg(not(target_arch = "wasm32"))]
    window::set_fullscreen(true);
    next_frame().await;
    let mut debugging: bool = true;
    let _left = &Player {
        up_key: KeyCode::W,
        down_key: KeyCode::S,
    };
    let _right = &Player {
        up_key: KeyCode::Up,
        down_key: KeyCode::Down,
    };
    let mut group = Group::new(POP_SIZE);
    group.inject_weights(&CHAMPION_SEED, ELITES, POP_SIZE);
    let mut former_champion = Individual::default();
    let mut current_champion = Individual::default();
    former_champion.inject_weights(&FIRST_CHAMPION);

    let start = get_time();
    let mut delta = get_time();
    let mut champion_game = Game::default();
    let mut score;

    for generation in 0..GENERATIONS {

        if is_key_released(KeyCode::Q) {
            println!(
                "Best weights: {:?}",
                group.individuals()[POP_SIZE - 1].weights()
            );
            quit();
        }

        if get_time() - delta < 0.1 {
            next_frame().await;
        }

        let longest_match = group.train();
        group.individuals_mut().sort();

        // Mini tournament to see if we are improving
        current_champion.inject_weights(group.individuals()[POP_SIZE - 1].weights());
        score = (0, 0);
        for j in 0..20 {
            let (winner, _) = champion_game.run_until(
                &mut former_champion,
                &mut current_champion,
                if j % 2 == 0 { Side::Right } else { Side::Left },
            );
            if winner == Side::Left {
                score.0 += 1;
            } else {
                score.1 += 1;
            }
        }

        draw_training(generation, start, get_time() - delta, longest_match, score);
        delta = get_time();
        group.mutate(ELITES, POP_SIZE);
    }

    println!(
        "Best weights: {:?}",
        group.individuals()[POP_SIZE - 1].weights()
    );

    let left = &former_champion;
    let right = &current_champion;
    let mut game = Game::default();

    loop {
        if is_key_released(KeyCode::Q) {
            quit();
        } else if is_key_released(KeyCode::N) {
            debugging = !debugging;
        }
        if get_time() - delta < 0.015 {
            continue;
        }

        clear_background(BLACK);
        draw_game(
            &debugging,
            screen_width(),
            screen_height(),
            &game,
            left,
            right,
        );
        game.tick(left, right);
        next_frame().await;
        delta = get_time();
    }
}
