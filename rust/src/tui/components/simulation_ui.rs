//! The simulation screen component.

use crate::{
    constants::{BALL_RADIUS, HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH, WIDTH},
    tui::app::{App, AppState},
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::Rect,
    style::Color,
    widgets::canvas::{Canvas, Rectangle},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
    Frame,
};

pub fn draw_simulation_ui(f: &mut Frame, app: &mut App, area: Rect) {
    let title = if let Some(sim) = &app.simulation {
        format!(
            "Simulation: '{}' (Left) vs '{}' (Right) | Press 'q' to quit",
            sim.p1_model_name,
            sim.p2_model_name.as_str()
        )
    } else {
        "Simulation Mode (Press 'q' to quit)".to_string()
    };

    let sim_block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = sim_block.inner(area);
    f.render_widget(sim_block.clone(), area);

    if let Some(sim) = &mut app.simulation {
        // State is updated in the main event loop, not here.
        let canvas = Canvas::default()
            .block(Block::default())
            .x_bounds([0.0, WIDTH as f64])
            .y_bounds([0.0, HEIGHT as f64])
            .paint(|ctx| {
                // Draw ball
                ctx.draw(&Rectangle {
                    x: (sim.game.ball_pos.0 - BALL_RADIUS) as f64,
                    y: (sim.game.ball_pos.1 - BALL_RADIUS) as f64,
                    width: (BALL_RADIUS * 2.0) as f64,
                    height: (BALL_RADIUS * 2.0) as f64,
                    color: Color::Yellow,
                });

                // Draw left paddle (paddle1)
                ctx.draw(&Rectangle {
                    x: 0.0,
                    y: (sim.game.paddle1_pos - PADDLE_HEIGHT as f32 / 2.0) as f64,
                    width: PADDLE_WIDTH as f64,
                    height: PADDLE_HEIGHT as f64,
                    color: Color::Blue,
                });

                // Draw right paddle (paddle2)
                ctx.draw(&Rectangle {
                    x: (WIDTH - PADDLE_WIDTH) as f64,
                    y: (sim.game.paddle2_pos - PADDLE_HEIGHT as f32 / 2.0) as f64,
                    width: PADDLE_WIDTH as f64,
                    height: PADDLE_HEIGHT as f64,
                    color: Color::Red,
                });
            });
        f.render_widget(canvas, inner_area);
    } else {
        let paragraph = Paragraph::new("No simulation running.")
            .block(sim_block)
            .wrap(Wrap { trim: true });
        f.render_widget(paragraph, area);
    }
}

/// Handles user input for the simulation screen.
pub fn handle_simulation_input(app: &mut App, key_code: KeyCode) {
    if let KeyCode::Char('q') = key_code {
        app.state = AppState::MainMenu;
        app.simulation = None; // Stop the simulation
    }
}
