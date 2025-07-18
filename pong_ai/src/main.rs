use std::{
    thread,
    time::{Duration, Instant},
};

use macroquad::{miniquad::window::quit, prelude::*};
use pong_ai::game::{
    // controllers::Player,
    state::{FRAME_RATE, Game, HEIGHT, PADDLE_LENGTH, PADDLE_WIDTH, WIDTH},
};
use pong_ai::nn::Group;

const POP_SIZE: usize = 2048;
const GENERATIONS: usize = 512;
const TOURNAMENT_SIZE: usize = POP_SIZE / 12;
const ELITES: usize = {
    POP_SIZE / 2
    // let mut i = 1;
    // while i * i < POP_SIZE {
    //     i += 1;
    // }
    // i
};

#[macroquad::main("MyGame")]
async fn main() {
    let mut debugging: bool = false;
    // let left = &Player {
    //     up_key: KeyCode::W,
    //     down_key: KeyCode::S,
    // };
    // let right = &Player {
    //     up_key: KeyCode::Up,
    //     down_key: KeyCode::Down,
    // };
    println!("ELITES: {}", ELITES);
    println!("POP_SIZE: {}", POP_SIZE);
    println!("GENERATIONS: {}", GENERATIONS);

    let mut group = Group::new(POP_SIZE);
    for i in 0..GENERATIONS {
        let start = Instant::now();
        group.train(TOURNAMENT_SIZE);
        group.individuals.sort();
        if i % 48 == 0 {
            println!("Generation: {}", i);
            println!("Population size: {}", POP_SIZE);
            println!("Tournament size: {}", TOURNAMENT_SIZE);
            println!("Matches done: {}", i * POP_SIZE * (TOURNAMENT_SIZE - 1));
            println!(
                "Matches per second: {}",
                (POP_SIZE * (TOURNAMENT_SIZE - 1)) as f64 / start.elapsed().as_secs_f64()
            );
            println!(
                "Fitness range: {} - {}",
                group.individuals[0].fitness,
                group.individuals[POP_SIZE - 1].fitness
            );
            println!(
                "Best weights: {:?}",
                group.individuals[POP_SIZE - 1].weights
            );
        }
        group.mutate(ELITES, POP_SIZE);
    }
    let left = &group.individuals[POP_SIZE - 1];
    let right = &group.individuals[POP_SIZE - 1];
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
        draw_game(&debugging, &game);
        game.tick(left, right);

        // Loop end
        next_frame().await;
        thread::sleep(Duration::from_millis((1000. / FRAME_RATE) as u64));
    }
}

fn draw_game(debugging: &bool, game: &Game) {
    // Set up aspect ratio based on screen size and game WIDTH/HEIGHT
    let aspect_ratio = WIDTH / HEIGHT;
    let screen_width = screen_width();
    let screen_height = screen_height();
    // Scale to whichever dimension is smaller
    let scale = if screen_width > screen_height * aspect_ratio {
        screen_height / HEIGHT
    } else {
        screen_width / WIDTH
    };
    // Created scaled values for the game elements
    let paddle_length = PADDLE_LENGTH * scale;
    let paddle_width = PADDLE_WIDTH * scale;
    let field_width = WIDTH * scale;
    let field_height = HEIGHT * scale;
    // Offset is the distance from the left edge of the screen to the left edge of the field
    let offset = (
        screen_width / 2. - field_width / 2.,
        screen_height / 2. - field_height / 2.,
    );

    draw_debug(
        debugging,
        game,
        aspect_ratio,
        scale,
        paddle_length,
        field_height,
        offset,
    );

    // Draw the field
    draw_rectangle_lines(offset.0, offset.1, field_width, field_height, 2., GREEN);
    draw_rectangle(
        offset.0,
        offset.1 + (game.left_pos() + 0.5) * scale - paddle_length / 2.,
        paddle_width,
        paddle_length,
        RED,
    );
    draw_rectangle(
        offset.0 + field_width - paddle_width,
        offset.1 + (game.right_pos() + 0.5) * scale - paddle_length / 2.,
        paddle_width,
        paddle_length,
        BLUE,
    );
    draw_circle(
        offset.0 + (game.ball_pos_x() + 0.5) * field_width,
        offset.1 + (game.ball_pos_y() + 0.5) * scale,
        0.01 * scale,
        GRAY,
    );
}

fn draw_debug(
    debugging: &bool,
    game: &Game,
    aspect_ratio: f32,
    scale: f32,
    paddle_length: f32,
    field_height: f32,
    offset: (f32, f32),
) {
    if *debugging {
        draw_fps();
        draw_text(&format!("State: {:?}", game.state), 0., 50., 16., YELLOW);
        draw_text(
            &format!(
                "Aspect: {}, Scale: {}, Offset: {}|{}",
                aspect_ratio, scale, offset.0, offset.1,
            ),
            0.,
            70.,
            16.,
            YELLOW,
        );
        draw_text(
            &format!(
                "Height: {}. Paddle length: {}. Paddle from: {} - {}",
                field_height,
                paddle_length,
                offset.1 + game.left_pos() * scale - paddle_length / 2.,
                offset.1 + game.left_pos() * scale + paddle_length / 2.
            ),
            0.,
            90.,
            16.,
            YELLOW,
        );
    }
}
