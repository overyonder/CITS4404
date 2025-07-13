//! The simulation screen component.

use crate::{
    config::Config,
    gamestate::constants::{PADDLE_HEIGHT, WIDTH},
    tui::{
        app::{App, AppState},
        simulation::SimulationState,
    },
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        canvas::{Canvas, Rectangle},
        Block, BorderType, Borders, Paragraph,
    },
    Frame,
};

/// Handles user input on the simulation screen.
pub fn handle_simulation_input(app: &mut App, key_code: KeyCode) {
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.state = AppState::MainMenu;
            app.simulation = None;
        }
        _ => {}
    }
}

/// Draws the simulation screen UI.
pub fn draw_simulation_ui(f: &mut Frame, app: &mut App, area: Rect) {
    if let Some(sim_state) = &app.simulation {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),       // Canvas
                Constraint::Length(6),    // Info panels
                Constraint::Length(6),    // Genome visualizations
            ])
            .split(area);

        draw_pong_canvas(f, sim_state, chunks[0]);
        draw_info_panels(f, sim_state, chunks[1]);
        draw_genome_panels(f, sim_state, chunks[2]);
    }
}

fn draw_pong_canvas(f: &mut Frame, sim_state: &SimulationState, area: Rect) {
    // Calculate proper aspect ratio based on game dimensions
    let game_width = WIDTH as f64;
            let game_height = crate::gamestate::constants::HEIGHT as f64;
    let game_aspect_ratio = game_width / game_height;
    
    // Calculate the maximum size we can use while maintaining aspect ratio
    let available_width = area.width as f64;
    let available_height = area.height as f64;
    let available_aspect_ratio = available_width / available_height;
    
    let (canvas_width, canvas_height) = if available_aspect_ratio > game_aspect_ratio {
        // Available area is wider than game - constrain by height
        let height = available_height;
        let width = height * game_aspect_ratio;
        (width as u16, height as u16)
    } else {
        // Available area is taller than game - constrain by width
        let width = available_width;
        let height = width / game_aspect_ratio;
        (width as u16, height as u16)
    };
    
    // Center the canvas in the available area
    let offset_x = (area.width.saturating_sub(canvas_width)) / 2;
    let offset_y = (area.height.saturating_sub(canvas_height)) / 2;
    
    let canvas_area = Rect {
        x: area.x + offset_x,
        y: area.y + offset_y,
        width: canvas_width,
        height: canvas_height,
    };

    let canvas = Canvas::default()
        .block(
            Block::default()
                .title("Pong Simulation")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .x_bounds([0.0, game_width])
        .y_bounds([0.0, game_height])
        .paint(|ctx| {
            // Draw paddles at field edges where collision detection expects them
            ctx.draw(&Rectangle {
                x: 0.0,  // Left paddle: at left edge of field
                y: sim_state.game_state.paddle1_pos as f64 - PADDLE_HEIGHT as f64 / 2.0,
                width: 2.0,
                height: PADDLE_HEIGHT as f64,
                color: Color::White,
            });
            ctx.draw(&Rectangle {
                x: game_width - 2.0,  // Right paddle: at right edge of field
                y: sim_state.game_state.paddle2_pos as f64 - PADDLE_HEIGHT as f64 / 2.0,
                width: 2.0,
                height: PADDLE_HEIGHT as f64,
                color: Color::White,
            });

            // Draw ball as a filled circle using multiple rectangles
            let ball_x = sim_state.game_state.ball_pos.0 as f64;
            let ball_y = sim_state.game_state.ball_pos.1 as f64;
            let radius = 2.0;
            
            // Draw a filled circle pattern
            for dx in -2..=2 {
                for dy in -2..=2 {
                    let distance = ((dx * dx + dy * dy) as f64).sqrt();
                    if distance <= radius {
                        ctx.draw(&Rectangle {
                            x: ball_x + dx as f64,
                            y: ball_y + dy as f64,
                            width: 1.0,
                            height: 1.0,
                            color: Color::Yellow,
                        });
                    }
                }
            }

            // Draw scores
            ctx.print(
                game_width / 4.0,
                game_height - 20.0,
                Line::from(sim_state.game_state.scores.0.to_string()).style(Style::default().fg(Color::Cyan)),
            );
            ctx.print(
                game_width * 3.0 / 4.0,
                game_height - 20.0,
                Line::from(sim_state.game_state.scores.1.to_string()).style(Style::default().fg(Color::Cyan)),
            );
        });
    f.render_widget(canvas, canvas_area);
}

fn draw_info_panels(f: &mut Frame, sim_state: &SimulationState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let left_info = create_info_text(&sim_state.left_config);
    let right_info = create_info_text(&sim_state.right_config);

    let p1_block = Block::default()
        .title("Left Player")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let p2_block = Block::default()
        .title("Right Player")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);

    let p1_paragraph = Paragraph::new(left_info).block(p1_block);
    let p2_paragraph = Paragraph::new(right_info).block(p2_block);

    f.render_widget(p1_paragraph, chunks[0]);
    f.render_widget(p2_paragraph, chunks[1]);
}

/// Helper function to create the metadata text for a player's info panel.
fn create_info_text(config: &Config) -> Text {
    let mut text = Text::default();
    let name_str = config
        .name
        .as_deref()
        .unwrap_or_else(|| "N/A");

    text.extend(vec![
        Line::from(vec![
            Span::styled("Model: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(name_str),
        ]),
        Line::from(vec![
            Span::styled("Engine: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(config.engine.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Activation: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(config.activation.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Trained for: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{} generations", config.generations)),
        ]),
    ]);
    text
}

fn draw_genome_panels(f: &mut Frame, sim_state: &SimulationState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    draw_genome_visualization(f, &sim_state.left_player.weights, "Left Player Genome", chunks[0]);
    draw_genome_visualization(f, &sim_state.right_player.weights, "Right Player Genome", chunks[1]);
}

fn draw_genome_visualization(f: &mut Frame, genome: &[f32], title: &str, area: Rect) {
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if genome.is_empty() {
        let placeholder = Paragraph::new("No genome data")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center);
        f.render_widget(placeholder, inner_area);
        return;
    }

    // Calculate grid dimensions to fit all weights
    let available_cells = (inner_area.width as usize) * (inner_area.height as usize);
    let weights_to_show = genome.len().min(available_cells);
    
    if weights_to_show == 0 {
        return;
    }

    // Create text representation using block characters
    let mut lines = Vec::new();
    let cols = inner_area.width as usize;
    
    // Normalize the genome weights to a 0-1 range for color mapping.
    let min_val = genome.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = genome.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = if max_val > min_val {
        max_val - min_val
    } else {
        1.0
    };

    for row_start in (0..weights_to_show).step_by(cols) {
        let mut spans = Vec::new();
        
        for col in 0..cols {
            let idx = row_start + col;
            if idx >= weights_to_show {
                // Fill remaining columns with spaces
                spans.push(Span::raw(" "));
                continue;
            }
            
            let weight = genome[idx];
            let normalized = (weight - min_val) / range;

            // Create a color gradient: blue -> cyan -> green -> yellow -> red
            let color = if normalized < 0.25 {
                // Blue to Cyan
                let t = normalized * 4.0;
                Color::Rgb(0, (t * 255.0) as u8, 255)
            } else if normalized < 0.5 {
                // Cyan to Green  
                let t = (normalized - 0.25) * 4.0;
                Color::Rgb(0, 255, ((1.0 - t) * 255.0) as u8)
            } else if normalized < 0.75 {
                // Green to Yellow
                let t = (normalized - 0.5) * 4.0;
                Color::Rgb((t * 255.0) as u8, 255, 0)
            } else {
                // Yellow to Red
                let t = (normalized - 0.75) * 4.0;
                Color::Rgb(255, ((1.0 - t) * 255.0) as u8, 0)
            };

            spans.push(Span::styled("█", Style::default().fg(color)));
        }
        
        lines.push(Line::from(spans));
        
        // Stop if we've filled the available height
        if lines.len() >= inner_area.height as usize {
            break;
        }
    }

    let genome_paragraph = Paragraph::new(lines);
    f.render_widget(genome_paragraph, inner_area);
}
