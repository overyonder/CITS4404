use macroquad::prelude::*;
use pong_ai::{game::{self, PADDLE_LENGTH}, nn};

#[macroquad::main("MyGame")]
async fn main() {
    let mut game = game::Game::new(
        game::Player {
            up_key: KeyCode::W,
            down_key: KeyCode::S,
        },
        game::Player {
            up_key: KeyCode::Up,
            down_key: KeyCode::Down,
        },
    );
    loop {
        clear_background(BLACK);
        draw_game(&game);
        game.tick();
        macroquad::time::draw_fps();
        next_frame().await;
        std::thread::sleep(std::time::Duration::from_millis((1000.0 / game::FRAME_RATE) as u64));
    }
}

fn draw_game(game: &game::Game) {
    // Set up aspect ratio based on screen size and game WIDTH/HEIGHT
    let aspect_ratio = game::WIDTH / game::HEIGHT;
    let screen_width = screen_width();
    let screen_height = screen_height();
    // Scale to whichever dimension is smaller
    let scale = if screen_width > screen_height * aspect_ratio {
        screen_height / game::HEIGHT
    } else {
        screen_width / game::WIDTH
    };
    // Created scaled values for the game elements
    let paddle_length = game::PADDLE_LENGTH * scale;
    let paddle_width = game::PADDLE_WIDTH * scale;
    let field_width = game::WIDTH * scale;
    let field_height = game::HEIGHT * scale;
    // Offset is the distance from the left edge of the screen to the left edge of the field
    let offset = (screen_width / 2.0 - field_width / 2.0, screen_height / 2.0 - field_height / 2.0);

    // Draw the field
    draw_rectangle_lines(offset.0, offset.1, field_width, field_height, 2.0, GREEN);
    draw_rectangle(
        // x, y, width, height, color
        // left player first, x is 0. then right player, x is 1.
        // y is inputs[0] and inputs[2]
        // scale to screen size
        offset.0,
        offset.1 + game.state[0] * scale - paddle_length,
        paddle_width,
        paddle_length,
        RED,
    );
    draw_rectangle(
        offset.0 + field_width - paddle_width,
        offset.1 + game.state[2] * scale - paddle_length,
        paddle_width,
        paddle_length,
        BLUE,
    );
    draw_circle(offset.0 + field_width / 2.0, offset.1 + field_height / 2.0, 0.01 * scale, GRAY);
}
