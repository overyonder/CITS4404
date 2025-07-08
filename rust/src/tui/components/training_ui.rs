//! The training screen component.

use crate::tui::{
    app::{App, AppState},
    training::{TrainingState},
};
use crossterm::event::KeyCode;


use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};

/// Handles user input on the training screen.
pub fn handle_training_input(app: &mut App, key_code: KeyCode) {
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => {
            // Check if training has been running for more than 30 seconds and save champion if so
            if let Some(training_state) = &app.training {
                let elapsed_time = training_state.start_time.elapsed().as_secs();
                if elapsed_time > 30 && !training_state.genome_weights.is_empty() {
                    save_champion_on_cancel(&training_state.genome_weights, &app.config);
                }
            }
            
            app.state = AppState::MainMenu;
            app.training = None; // Clean up training state
            app.tx = None;
            app.rx = None;
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

/// Saves the champion genome when training is cancelled after 30+ seconds
fn save_champion_on_cancel(genome_weights: &[f32], config: &crate::config::Config) {
    use chrono::Utc;
    use std::fs;
    use std::path::Path;
    
    // Create a timestamped filename for the cancelled training session
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("cancelled_champion_{}.json", timestamp);
    let save_path = Path::new("models").join(&filename);
    
    // Create the models directory if it doesn't exist
    if let Some(parent_dir) = save_path.parent() {
        if let Err(e) = fs::create_dir_all(parent_dir) {
            tracing::error!("Failed to create models directory: {}", e);
            return;
        }
    }
    
    // Create a config for saving with current timestamp
    let mut save_config = config.clone();
    save_config.name = Some(format!("Cancelled Training {}", timestamp));
    save_config.date_trained = Some(Utc::now());
    
    // Save the genome using a simple JSON format
    let save_data = serde_json::json!({
        "config": save_config,
        "weights": genome_weights
    });
    
    match serde_json::to_string_pretty(&save_data) {
        Ok(json_string) => {
            if let Err(e) = fs::write(&save_path, json_string) {
                tracing::error!("Failed to save champion genome: {}", e);
            } else {
                tracing::info!("Champion genome saved to: {}", save_path.display());
            }
        }
        Err(e) => {
            tracing::error!("Failed to serialize champion genome: {}", e);
        }
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

        // Main area layout: charts and fitness history
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(&[
                Constraint::Length(8),     // Training rate chart
                Constraint::Length(8),     // Improvement rate chart  
                Constraint::Length(8),     // Fitness history
            ])
            .split(main_area);

        // Draw new charts above fitness history
        draw_training_rate_chart(f, training_state, main_chunks[0]);
        draw_improvement_rate_chart(f, training_state, main_chunks[1]);

        // Fitness History at bottom of main panel
        draw_fitness_history(f, training_state, main_chunks[2]);

        // Sidebar layout: Progress, Champion Genome, Info
        let sidebar_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),  // Progress (smaller now)
                Constraint::Min(8),     // Champion Genome 
                Constraint::Length(20), // Info (much taller to show settings + metrics)
            ])
            .split(sidebar_area);

        // Progress Section
        draw_progress_section(f, training_state, sidebar_chunks[0]);

        // Champion Genome Visualization
        draw_champion_genome(f, training_state, sidebar_chunks[1]);

        // Enhanced Info Panel with training metrics
        draw_enhanced_info_panel(f, app, sidebar_chunks[2]);
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
        Line::from(format!("Concurrent: {}", app.config.concurrent)),
        Line::from(format!("Elite Count: {}", app.config.elite_count)),
        Line::from(format!("Reproduction: {}", app.config.reproduction_strategy)),
        Line::from(format!("Mutation: {}", app.config.mutation_strategy)),
    ];

    // Only show mutation parameters when using Modern mutation strategy
    if matches!(app.config.mutation_strategy, crate::config::MutationStrategy::Modern) {
        lines.insert(lines.len() - 1, Line::from(format!("Mutation Rate: {:.3}", app.config.mutation_rate)));
        lines.insert(lines.len() - 1, Line::from(format!("Mutation Str: {:.3}", app.config.mutation_strength)));
    }

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

    // Draw fitness scale on the right side
    let scale_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(6)])
        .split(inner_area);

    let sparkline = Sparkline::default()
        .data(&fitness_data)
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(sparkline, scale_chunks[0]);

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

fn draw_training_rate_chart(f: &mut Frame, training_state: &TrainingState, area: Rect) {
    let block = Block::default()
        .title("Training Rate (gen/s)")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if training_state.training_rate_history.is_empty() || inner_area.area() == 0 {
        return;
    }

    // Scale training rate data for sparkline
    let max_rate = training_state.training_rate_history.iter().fold(0.0f32, |a, &b| a.max(b));
    let rate_data: Vec<u64> = if max_rate > 0.0 {
        training_state
            .training_rate_history
            .iter()
            .map(|&rate| ((rate / max_rate) * 100.0) as u64)
            .collect()
    } else {
        vec![0; training_state.training_rate_history.len()]
    };

    // Draw with scale on the right
    let scale_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(6)])
        .split(inner_area);

    let sparkline = Sparkline::default()
        .data(&rate_data)
        .style(Style::default().fg(Color::Green));
    f.render_widget(sparkline, scale_chunks[0]);

    if scale_chunks[1].height >= 3 {
        let _current_rate = training_state.get_current_training_rate();
        let scale_text = format!("{:.2}\n\n0.00", max_rate);
        let scale_paragraph = Paragraph::new(scale_text)
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Right);
        f.render_widget(scale_paragraph, scale_chunks[1]);
    }
}

fn draw_improvement_rate_chart(f: &mut Frame, training_state: &TrainingState, area: Rect) {
    let block = Block::default()
        .title("Improvement Rate (fit/s)")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if training_state.improvement_rate_history.is_empty() || inner_area.area() == 0 {
        return;
    }

    // Scale improvement rate data for sparkline
    let max_improvement = training_state.improvement_rate_history.iter().fold(0.0f32, |a, &b| a.max(b));
    let improvement_data: Vec<u64> = if max_improvement > 0.0 {
        training_state
            .improvement_rate_history
            .iter()
            .map(|&rate| ((rate / max_improvement) * 100.0) as u64)
            .collect()
    } else {
        vec![0; training_state.improvement_rate_history.len()]
    };

    // Draw with scale on the right
    let scale_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(6)])
        .split(inner_area);

    let sparkline = Sparkline::default()
        .data(&improvement_data)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(sparkline, scale_chunks[0]);

    if scale_chunks[1].height >= 3 {
        let scale_text = format!("{:.3}\n\n0.000", max_improvement);
        let scale_paragraph = Paragraph::new(scale_text)
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Right);
        f.render_widget(scale_paragraph, scale_chunks[1]);
    }
}

fn draw_enhanced_info_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title("Configuration & Metrics")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    let engine_color = match app.config.engine.to_string().as_str() {
        "CPU" => Color::Gray,
        "GPU" => Color::Magenta,
        _ => Color::White,
    };

    let activation_color = match app.config.activation {
        crate::config::Activation::ClampedLinear => Color::Rgb(128, 128, 128),
        crate::config::Activation::Tanh => Color::Cyan,
        crate::config::Activation::Relu => Color::Green,
        crate::config::Activation::Atan => Color::Yellow,
        crate::config::Activation::Linear => Color::White,
        crate::config::Activation::Sigmoid => Color::Rgb(255, 165, 0),
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Configuration:", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Engine: "),
            Span::styled(
                format!(" {} ", app.config.engine.to_string().to_uppercase()),
                Style::default().bg(engine_color).fg(Color::Black),
            ),
        ]),
        Line::from(vec![
            Span::raw("Activation: "),
            Span::styled(
                format!(" {} ", app.config.activation.to_string()),
                Style::default().bg(activation_color).fg(Color::Black),
            ),
        ]),
        Line::from(format!("Population: {}", app.config.population_size)),
        Line::from(format!("Elite Count: {}", app.config.elite_count)),
        Line::from(format!("Generations: {}", app.config.generations)),
        Line::from(format!("Fitness: {}", app.config.fitness_func)),
        Line::from(format!("Reproduction: {}", app.config.reproduction_strategy)),
        Line::from(format!("Mutation: {}", app.config.mutation_strategy)),
        Line::from(format!("Concurrent: {}", app.config.concurrent)),
    ];

    // Add training metrics if training is active
    if let Some(training_state) = &app.training {
        lines.extend_from_slice(&[
            Line::from(""),
            Line::from(vec![
                Span::styled("Live Metrics:", Style::default().fg(Color::Yellow)),
            ]),
            Line::from(format!("Matches: {}", training_state.total_matches_simulated)),
            Line::from(format!("Rate: {:.2} gen/s", training_state.get_current_training_rate())),
            Line::from(format!("Improve: {:.3} fit/s", training_state.get_current_improvement_rate())),
            Line::from(format!("Max Score: {}", training_state.get_max_possible_score())),
        ]);
    }

    let info_paragraph = Paragraph::new(lines).alignment(Alignment::Left);
    f.render_widget(info_paragraph, inner_area);
}


