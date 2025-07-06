//! The training screen component.

use crate::tui::{
    app::{App, AppState, Tab},
    training::{GenerationState, MatchupState, TrainingState},
};
use crossterm::event::KeyCode;
use ratatui::widgets::canvas::{Canvas, Rectangle};

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, Paragraph, Sparkline, Tabs},
    Frame,
};
use tui_logger::TuiLoggerWidget;

/// Handles user input on the training screen.
pub fn handle_training_input(app: &mut App, key_code: KeyCode) {
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.state = AppState::MainMenu;
            app.training = None; // Clean up training state
            app.tx = None;
            app.rx = None;
        }
        KeyCode::Tab => {
            app.next_tab();
        }
        _ => {}
    }
}

pub fn draw_training_ui(f: &mut Frame, app: &mut App, area: Rect) {
    if let Some(training_state) = &mut app.training {
        // Create a two-column layout
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(&[Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        let main_area = chunks[0];
        let sidebar_area = chunks[1];

        // Main area layout (tabs and content)
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(&[Constraint::Length(3), Constraint::Min(0)])
            .split(main_area);

        let titles: Vec<Line<'_>> = ["Generations", "Matchups"]
            .iter()
            .map(|t| Line::from(Span::styled(*t, Style::default())))
            .collect();

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .title("View")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .select(app.active_tab as usize)
            .style(Style::default().fg(Color::Cyan))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));

        f.render_widget(tabs, main_chunks[0]);

        match app.active_tab {
            Tab::Generations => draw_generations_view(f, training_state, main_chunks[1]),
            Tab::Matchups => draw_matchups_view(f, training_state, main_chunks[1]),
        }

        // Sidebar layout
        let sidebar_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Progress
                Constraint::Length(5), // Fitness History
                Constraint::Min(10),   // Champion Genome
                Constraint::Length(5), // Info
                Constraint::Min(5),    // Log
            ])
            .split(sidebar_area);

        // Progress Section
        let progress_block = Block::default()
            .title("Progress")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded);
        let progress_area = sidebar_chunks[0];
        f.render_widget(progress_block, progress_area);

        let progress_chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints(&[
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .split(progress_area);

        // Progress Gauge
        let progress_percentage = if training_state.total_generations > 0 {
            (training_state.current_generation as f64 / training_state.total_generations as f64)
                .min(1.0)
        } else {
            0.0
        };
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .percent((progress_percentage * 100.0) as u16);
        f.render_widget(gauge, progress_chunks[0]);

        // Generations Text
        let generations_text = format!(
            "Gen: {} / {}",
            training_state.current_generation, training_state.total_generations
        );
        let text_paragraph = Paragraph::new(generations_text).alignment(Alignment::Center);
        f.render_widget(text_paragraph, progress_chunks[1]);

        // Stopwatch
        let elapsed = training_state.start_time.elapsed();
        let elapsed_secs = elapsed.as_secs();
        let stopwatch_text = format!("Time: {:02}:{:02}", elapsed_secs / 60, elapsed_secs % 60);
        let stopwatch_paragraph = Paragraph::new(stopwatch_text).alignment(Alignment::Center);
        f.render_widget(stopwatch_paragraph, progress_chunks[2]);

        // Fitness History Sparkline
        let fitness_data: Vec<u64> = training_state
            .fitness_history
            .iter()
            .map(|&f| f as u64)
            .collect();

        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .title("Fitness History")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .data(&fitness_data)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(sparkline, sidebar_chunks[1]);

        // Champion Genome Visualization
        let (cols, rows) = (31, 7); // 31x7 grid for 217 weights
        let cell_width = 1.2;
        let cell_height = 1.2;
        let champion_canvas = Canvas::default()
            .block(
                Block::default()
                    .title("Champion Genome")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .x_bounds([0.0, cols as f64 * cell_width])
            .y_bounds([0.0, rows as f64 * cell_height])
            .paint(|ctx| {
                let genome = &training_state.genome_weights;
                if !genome.is_empty() {
                    // Normalize the genome weights to a 0-1 range for color mapping.
                    let min_val = genome.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max_val = genome.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let range = if max_val > min_val {
                        max_val - min_val
                    } else {
                        1.0
                    };

                    for (i, &weight) in genome.iter().enumerate() {
                        let normalized = (weight - min_val) / range;

                        // Map normalized value to a blue-to-red color gradient.
                        let r = (normalized * 255.0) as u8;
                        let b = 255 - r;
                        let color = Color::Rgb(r, 0, b);

                        let col = i % cols;
                        let row = i / cols;

                        ctx.draw(&Rectangle {
                            x: col as f64 * cell_width,
                            y: (rows - 1 - row) as f64 * cell_height, // Invert Y-axis
                            width: 1.0,
                            height: 1.0,
                            color,
                        });
                    }
                }
            });
        f.render_widget(champion_canvas, sidebar_chunks[2]);

        // Info Panel
        draw_info_panel(f, app, sidebar_chunks[3]);

        // Log Panel
        let log_widget: TuiLoggerWidget = TuiLoggerWidget::default()
            .block(
                Block::default()
                    .title("Log")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .style_error(Style::default().fg(Color::Red))
            .style_debug(Style::default().fg(Color::Green))
            .style_warn(Style::default().fg(Color::Yellow))
            .style_trace(Style::default().fg(Color::Magenta))
            .style_info(Style::default().fg(Color::Cyan))
            .output_separator(' ');
        f.render_widget(log_widget, sidebar_chunks[4]);
    }
}

/// Draws the content for the "Generations" tab.
fn draw_generations_view(f: &mut Frame, training_state: &TrainingState, area: Rect) {
    let block = Block::default()
        .title("Generations Progress")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if training_state.generations.is_empty() || inner_area.area() == 0 {
        return;
    }

    let num_generations = training_state.generations.len();
    let (cols, rows) = calculate_grid_size(num_generations, inner_area.width as usize);

    if cols == 0 || rows == 0 {
        return;
    }

    // Use a slightly larger cell size for visibility
    let cell_width = 2.0;
    let cell_height = 1.0;

    let canvas = Canvas::default()
        .block(Block::default()) // No inner block
        .x_bounds([0.0, cols as f64 * cell_width])
        .y_bounds([0.0, rows as f64 * cell_height])
        .paint(move |ctx| {
            for (i, state) in training_state.generations.iter().enumerate() {
                let col = i % cols;
                let row = i / cols;

                let color = match state {
                    GenerationState::Pending => Color::DarkGray,
                    GenerationState::InProgress => Color::Yellow,
                    GenerationState::Completed => Color::Green,
                };

                ctx.draw(&Rectangle {
                    x: col as f64 * cell_width,
                    y: (rows - 1 - row) as f64 * cell_height, // Invert Y-axis
                    width: cell_width - 0.5,                  // Add some spacing
                    height: cell_height,
                    color,
                });
            }
        });

    f.render_widget(canvas, inner_area);
}

/// Draws the content for the "Matchups" tab, showing a grid of game states.
fn draw_matchups_view(f: &mut Frame, training_state: &TrainingState, area: Rect) {
    let block = Block::default()
        .title("Matchups Grid")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if training_state.matchups.is_empty() || inner_area.area() == 0 {
        return;
    }

    let num_matchups = training_state.matchups.len();
    let (cols, rows) = calculate_grid_size(num_matchups, inner_area.width as usize);

    if cols == 0 || rows == 0 {
        return;
    }

    let cell_size = 1.2;
    let canvas = Canvas::default()
        .x_bounds([0.0, cols as f64 * cell_size])
        .y_bounds([0.0, rows as f64 * cell_size])
        .paint(move |ctx| {
            for (i, state) in training_state.matchups.iter().enumerate() {
                let col = i % cols;
                let row = i / cols;

                let color = match state {
                    MatchupState::Pending => Color::DarkGray,
                    MatchupState::InProgress => Color::Yellow,
                    MatchupState::Completed => Color::Green,
                };

                ctx.draw(&Rectangle {
                    x: col as f64 * cell_size,
                    y: (rows - 1 - row) as f64 * cell_size,
                    width: 1.0,
                    height: 1.0,
                    color,
                });
            }
        });

    f.render_widget(canvas, inner_area);
}

/// Draws the info panel with colorful badges for configuration parameters.
fn draw_info_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title("Info")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    let (engine_str, is_concurrent) = {
        let engine_name = app.config.engine.to_string();
        if let Some(stripped) = engine_name.strip_prefix("concurrent-") {
            (stripped.to_string(), true)
        } else {
            (engine_name, false)
        }
    };

    let engine_color = match engine_str.as_str() {
        "stack" => Color::Gray,
        "heap" => Color::Red,
        "simd" => Color::Blue,
        "gpu" => Color::Magenta,
        _ => Color::White,
    };

    let activation_color = match app.config.activation {
        crate::config::Activation::Tanh => Color::Cyan,
        crate::config::Activation::Relu => Color::Green,
        crate::config::Activation::Atan => Color::Yellow,
        crate::config::Activation::Linear => Color::White,
        crate::config::Activation::Sigmoid => Color::Rgb(255, 165, 0), // Orange
    };

    let mut lines = vec![
        Line::from(vec![
            Span::raw("Engine: "),
            Span::styled(
                format!(" {} ", engine_str.to_uppercase()),
                Style::default().bg(engine_color).fg(Color::Black),
            ),
        ]),
        Line::from(vec![
            Span::raw("Activation: "),
            Span::styled(
                format!(" {} ", app.config.activation.to_string().to_uppercase()),
                Style::default().bg(activation_color).fg(Color::Black),
            ),
        ]),
        Line::from(format!("Fitness Fn: {}", app.config.fitness_func)),
        Line::from(format!("Population: {}", app.config.population_size)),
        Line::from(format!("Mutation Rate: {}", app.config.mutation_rate)),
        Line::from(format!(
            "Mutation Strength: {}",
            app.config.mutation_strength
        )),
    ];

    if is_concurrent {
        lines.insert(
            1,
            Line::from(vec![
                Span::raw("Mode: "),
                Span::styled(
                    " CONCURRENT ",
                    Style::default().bg(Color::LightBlue).fg(Color::Black),
                ),
            ]),
        );
    }

    let info_paragraph = Paragraph::new(lines).alignment(Alignment::Left);
    f.render_widget(info_paragraph, inner_area);
}

fn calculate_grid_size(num_items: usize, area_width: usize) -> (usize, usize) {
    if num_items == 0 {
        return (0, 0);
    }
    const CELL_WIDTH: usize = 2; // Each block is at least 2 chars wide
    let max_cols = (area_width / CELL_WIDTH).max(1);
    let cols = (num_items as f64).sqrt().ceil() as usize;
    let cols = cols.min(max_cols);
    let rows = (num_items + cols - 1) / cols;
    (cols, rows)
}
