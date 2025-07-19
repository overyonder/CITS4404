use crate::game::{
    controllers::Controller,
    state::{Game, HEIGHT, PADDLE_LENGTH, PADDLE_WIDTH, WIDTH},
};
use macroquad::prelude::*;

pub fn draw_game<T: Controller, U: Controller>(
    debugging: &bool,
    screen_width: f32,
    screen_height: f32,
    game: &Game,
    left: &T,
    right: &U,
) {
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

    draw_debug(debugging, aspect_ratio, scale, offset, game, left, right);

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

fn draw_debug<T: Controller, U: Controller>(
    debugging: &bool,
    aspect_ratio: f32,
    scale: f32,
    offset: (f32, f32),
    game: &Game,
    left: &T,
    right: &U,
) {
    if *debugging {
        draw_fps();
        draw_text(
            &format!(
                "Aspect: {}, Scale: {}, Offset: {}|{}",
                aspect_ratio, scale, offset.0, offset.1,
            ),
            0.,
            50.,
            16.,
            YELLOW,
        );
        draw_state(offset, game, left, right);
    } else {
        draw_text("Press N to toggle debug info", 0., 0., 16., YELLOW);
    }
}

fn draw_genome<T: Controller>(offset: (f32, f32), controller: &T, game: &Game) -> Option<()> {
    let activations = controller.genome()?.activations(&game.state);
    let (layer2, layer3, output) = activations;
    for i in 0..16 {
        let color = layer2[i].to_color();
        draw_rectangle(offset.0 + i as f32 * 10., offset.1, 10., 10., color);
    }
    for i in 0..4 {
        let color = layer3[i].to_color();
        draw_rectangle(offset.0 + i as f32 * 10., offset.1 + 10., 10., 10., color);
    }
    let color = output.to_color();
    draw_rectangle(offset.0 + 4 as f32 * 10., offset.1 + 10., 10., 10., color);
    Some(())
}

fn draw_state<T: Controller, U: Controller>(offset: (f32, f32), game: &Game, left: &T, right: &U) {
    draw_genome(offset, left, game);
    draw_genome(offset, right, game);
}

trait F32Ext {
    fn to_color(self) -> Color;
}

impl F32Ext for f32 {
    /// -1 is blue, 0 is white, 1 is red
    fn to_color(self) -> Color {
        let r = self.clamp(0., 1.);
        let g = (1. - r).clamp(0., 1.);
        let b = (1. - r - g).clamp(0., 1.);
        Color::new(r, g, b, 1.)
    }
}
