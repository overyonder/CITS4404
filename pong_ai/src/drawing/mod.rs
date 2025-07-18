use macroquad::prelude::*;
use crate::game::state::{Game, HEIGHT, PADDLE_LENGTH, PADDLE_WIDTH, WIDTH};

pub fn draw_game(debugging: &bool, game: &Game, screen_width: f32, screen_height: f32) {
    // Set up aspect ratio based on screen size and game WIDTH/HEIGHT
    let aspect_ratio = WIDTH / HEIGHT;
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
