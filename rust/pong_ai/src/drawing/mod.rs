use crate::{
    constants::{FONT_SIZE, GENERATIONS, POP_SIZE, ELITES, TOURNAMENT_SIZE},
    game::{
        controllers::Controller,
        state::{FRAME_RATE, Game, HEIGHT, PADDLE_LENGTH, PADDLE_WIDTH, Side, WIDTH},
    },
};
use macroquad::prelude::*;
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

pub fn draw_training(
    generation: usize,
    start_time: f64,
    delta: f64,
    longest_match: usize,
    champions_score: (i16, i16),
) {
    draw_text("Loading...", 0., FONT_SIZE * 0.5, FONT_SIZE, YELLOW);
    draw_text(
        &format!("Generations: {GENERATIONS}"),
        0.,
        FONT_SIZE * 1.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!("Generations complete: {}", (generation + 1)),
        0.,
        FONT_SIZE * 2.5,
        FONT_SIZE,
        GREEN,
    );
    for j in generation..GENERATIONS {
        draw_line(
            j as f32 * screen_width() * 0.9 / GENERATIONS as f32,
            FONT_SIZE * 3.5,
            j as f32 * screen_width() * 0.9 / GENERATIONS as f32,
            FONT_SIZE * 4.5,
            screen_width() * 0.9 / GENERATIONS as f32,
            WHITE,
        );
    }
    for j in 0..generation {
        draw_line(
            j as f32 * screen_width() * 0.9 / GENERATIONS as f32,
            FONT_SIZE * 3.5,
            j as f32 * screen_width() * 0.9 / GENERATIONS as f32,
            FONT_SIZE * 4.5,
            screen_width() * 0.9 / GENERATIONS as f32,
            GREEN,
        );
    }
    draw_text(
        &format!("Population size: {POP_SIZE}"),
        0.,
        FONT_SIZE * 5.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!("Elites: {ELITES}"),
        0.,
        FONT_SIZE * 6.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!("Tournament size: {TOURNAMENT_SIZE}"),
        0.,
        FONT_SIZE * 7.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!(
            "Matches done: {}",
            (generation + 1) * POP_SIZE * (TOURNAMENT_SIZE - 1)
        ),
        0.,
        FONT_SIZE * 8.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!("Elapsed time: {:?}", get_time() - start_time),
        0.,
        FONT_SIZE * 9.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!("Round time: {delta:?}"),
        0.,
        FONT_SIZE * 10.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!(
            "Longest match last generation: {} ticks of possible {} ({} seconds)",
            longest_match,
            FRAME_RATE * 120.,
            120
        ),
        0.,
        FONT_SIZE * 11.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!(
            "Matches per generation: {}",
            POP_SIZE * (TOURNAMENT_SIZE - 1)
        ),
        0.,
        FONT_SIZE * 12.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!(
            "Matches per second: {}",
            (POP_SIZE * (TOURNAMENT_SIZE - 1)) as f64 / delta
        ),
        0.,
        FONT_SIZE * 13.5,
        FONT_SIZE,
        WHITE,
    );
    draw_text(
        &format!(
            "Versus former champion: {} - {}",
            champions_score.0, champions_score.1
        ),
        0.,
        FONT_SIZE * 14.5,
        FONT_SIZE,
        PINK,
    );
}

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

    draw_debug(
        debugging,
        aspect_ratio,
        scale,
        offset,
        (field_width, field_height),
        game,
        left,
        right,
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

fn draw_debug<T: Controller, U: Controller>(
    debugging: &bool,
    aspect_ratio: f32,
    scale: f32,
    offset: (f32, f32),
    field_size: (f32, f32),
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
        draw_state(offset, field_size, game, left, right);
    } else {
        draw_text("Press N to toggle debug info", 0., 0., 16., YELLOW);
    }
}

fn draw_genome<T: Controller>(
    offset: (f32, f32),
    field_size: (f32, f32),
    side: Side,
    controller: &T,
    game: &Game,
) -> Option<()> {
    let activations = controller.genome()?.activations(&game.state);
    let (layer2, layer3, output) = activations;

    let start = match side {
        Side::Left => offset.0,
        Side::Right => offset.0 + field_size.0 - 10.,
        Side::Neither => return None,
    };

    let direction = match side {
        Side::Left => 1.,
        Side::Right => -1.,
        Side::Neither => return None,
    };

    for i in 0..16 {
        let i_f32 = i as f32;
        let color = layer2[i].to_color();
        draw_rectangle(start + direction * i_f32 * 10., offset.1, 10., 10., color);
    }
    for i in 0..4 {
        let i_f32 = i as f32;
        let color = layer3[i].to_color();
        draw_rectangle(
            start + direction * (i_f32 + 6.) * 10.,
            offset.1 + 10.,
            10.,
            10.,
            color,
        );
    }
    let color = output.to_color();
    draw_rectangle(
        start + direction * 7.5 * 10.,
        offset.1 + 20.,
        10.,
        10.,
        color,
    );
    Some(())
}

fn draw_state<T: Controller, U: Controller>(
    offset: (f32, f32),
    field_size: (f32, f32),
    game: &Game,
    left: &T,
    right: &U,
) {
    draw_genome(offset, field_size, Side::Left, left, game);
    draw_genome(offset, field_size, Side::Right, right, game);
}
