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
        KeyCode::Enter => {
            if let Some(ts) = &app.training {
                if !ts.running {
                    app.state = AppState::MainMenu;
                    app.training = None;
                    app.tx = None;
                    app.rx = None;
                }
            }
        }
        _ => {}
    }
}

pub fn draw_training_ui(f: &mut Frame, app: &mut App, area: Rect) {
    // Check for finished state first to avoid borrow checker issues.
    if let Some(training_state) = &app.training {
        if !training_state.running {
            draw_finished_popup(f, area);
            return;
        }
    }

    if let Some(training_state) = &mut app.training {
        // Create a two-column layout
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(&[Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        let main_area = chunks[0];
        let sidebar_area = chunks[1];

        // Main area layout: tabs, content, and fitness history at bottom
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(&[
                Constraint::Length(3),     // Tabs
                Constraint::Min(10),       // Content area
                Constraint::Length(8),     // Fitness history
            ])
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

        // Fitness History at bottom of main panel
        draw_fitness_history(f, training_state, main_chunks[2]);

        // Sidebar layout: Progress, Champion Genome, Info
        let sidebar_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6),  // Progress 
                Constraint::Min(8),     // Champion Genome (larger)
                Constraint::Length(12), // Info (larger to prevent clipping)
            ])
            .split(sidebar_area);

        // Progress Section
        draw_progress_section(f, training_state, sidebar_chunks[0]);

        // Champion Genome Visualization
        draw_champion_genome(f, training_state, sidebar_chunks[1]);

        // Info Panel
        draw_info_panel(f, app, sidebar_chunks[2]);
    }
}

fn draw_finished_popup(f: &mut Frame, area: Rect) {
    let popup_text = "Training Finished!\n\nPress 'Enter' to return to the main menu.";
    let block = Block::default()
        .title("Status")
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(Color::Green));

    let text = Paragraph::new(popup_text)
        .block(block)
        .alignment(Alignment::Center);

    // Center the popup in the middle of the screen
    let popup_area = centered_rect(50, 20, area);
    f.render_widget(ratatui::widgets::Clear, popup_area); // Clear the area behind the popup
    f.render_widget(text, popup_area);
}

/// Helper to create a centered rectangle for popups.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
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

    // Ensure the grid fits within the available area
    let cell_size = ((inner_area.width as f64) / cols as f64).min(
        (inner_area.height as f64) / rows as f64
    ).max(1.0);
    
    let canvas = Canvas::default()
        .x_bounds([0.0, cols as f64 * cell_size])
        .y_bounds([0.0, rows as f64 * cell_size])
        .paint(move |ctx| {
            for (i, state) in training_state.matchups.iter().enumerate() {
                if i >= cols * rows {
                    break; // Don't draw more items than fit in the grid
                }
                
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
                    width: cell_size * 0.8, // Add some spacing between cells
                    height: cell_size * 0.8,
                    color,
                });
            }
        });

    f.render_widget(canvas, inner_area);
}

/// Draws the info panel with colorful badges for configuration parameters.
fn draw_info_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title("Configuration")
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
        crate::config::Activation::ClampedLinear => Color::Rgb(128, 128, 128), // Gray
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
        Line::from(format!("Fitness: {}", app.config.fitness_func)),
        Line::from(format!("Population: {}", app.config.population_size)),
        Line::from(format!("Generations: {}", app.config.generations)),
        Line::from(format!("Mutation Rate: {:.3}", app.config.mutation_rate)),
        Line::from(format!("Mutation Str: {:.3}", app.config.mutation_strength)),
        Line::from(format!("Concurrent: {}", app.config.concurrent)),
        Line::from(format!("Elite Count: {}", app.config.elite_count)),
        Line::from(format!("Reproduction: {}", app.config.reproduction_strategy)),
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

fn draw_progress_section(f: &mut Frame, training_state: &TrainingState, area: Rect) {
    let progress_block = Block::default()
        .title("Progress")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    f.render_widget(progress_block, area);

    let progress_chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(&[
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

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

    // Current fitness
    if let Some(last_fitness) = training_state.fitness_history.last() {
        let fitness_text = format!("Best: {:.3}", last_fitness);
        let fitness_paragraph = Paragraph::new(fitness_text).alignment(Alignment::Center);
        f.render_widget(fitness_paragraph, progress_chunks[3]);
    }
}

fn draw_fitness_history(f: &mut Frame, training_state: &TrainingState, area: Rect) {
    let block = Block::default()
        .title("Fitness History")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if training_state.fitness_history.is_empty() || inner_area.area() == 0 {
        return;
    }

    // Get fitness bounds for scaling
    let min_fitness = training_state.fitness_history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_fitness = training_state.fitness_history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let fitness_range = if max_fitness > min_fitness { max_fitness - min_fitness } else { 1.0 };

    // Calculate how many data points we can show
    let max_points = inner_area.width as usize;
    let fitness_data: Vec<u64> = if training_state.fitness_history.len() <= max_points {
        // Show all data
        training_state
            .fitness_history
            .iter()
            .map(|&f| (((f - min_fitness) / fitness_range) * 100.0) as u64)
            .collect()
    } else {
        // Sample the data to fit
        let step = training_state.fitness_history.len() as f64 / max_points as f64;
        (0..max_points)
            .map(|i| {
                let idx = (i as f64 * step) as usize;
                let f = training_state.fitness_history[idx];
                (((f - min_fitness) / fitness_range) * 100.0) as u64
            })
            .collect()
    };

    let sparkline = Sparkline::default()
        .data(&fitness_data)
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(sparkline, inner_area);

    // Draw fitness scale on the right side
    let scale_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(8)])
        .split(inner_area);

    if scale_chunks[1].height >= 4 {
        let scale_text = format!("Max\n{:.3}\n\nMin\n{:.3}", max_fitness, min_fitness);
        let scale_paragraph = Paragraph::new(scale_text)
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Right);
        f.render_widget(scale_paragraph, scale_chunks[1]);
    } else if scale_chunks[1].height >= 3 {
        let scale_text = format!("{:.3}\n\n{:.3}", max_fitness, min_fitness);
        let scale_paragraph = Paragraph::new(scale_text)
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Right);
        f.render_widget(scale_paragraph, scale_chunks[1]);
    }
}

fn draw_champion_genome(f: &mut Frame, training_state: &TrainingState, area: Rect) {
    let title = if let Some(last_fitness) = training_state.fitness_history.last() {
        format!("Champion Genome (Fitness: {:.3})", last_fitness)
    } else {
        "Champion Genome".to_string()
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    let genome = &training_state.genome_weights;
    if genome.is_empty() {
        // Show placeholder text if no genome data
        let placeholder = Paragraph::new("No genome data available")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center);
        f.render_widget(placeholder, inner_area);
        return;
    }

    let champion_canvas = Canvas::default()
        .x_bounds([0.0, 31.0])
        .y_bounds([0.0, 7.0])
        .paint(|ctx| {
            // Normalize the genome weights to a 0-1 range for color mapping.
            let min_val = genome.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = genome.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = if max_val > min_val {
                max_val - min_val
            } else {
                1.0
            };

            for (i, &weight) in genome.iter().take(217).enumerate() {
                let normalized = (weight - min_val) / range;

                // Create a more visible color gradient: blue -> cyan -> green -> yellow -> red
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

                let col = i % 31;
                let row = i / 31;

                ctx.draw(&Rectangle {
                    x: col as f64,
                    y: (6 - row) as f64, // Invert Y-axis
                    width: 1.0,
                    height: 1.0,
                    color,
                });
            }
        });
    f.render_widget(champion_canvas, inner_area);
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
