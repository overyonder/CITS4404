//! Enhanced training configuration screen with comprehensive parameter editing and validation.
//!
//! # Teaching Note: Configuration UI Design
//! This component demonstrates effective parameter editing interface design:
//! - Real-time validation and feedback
//! - Clear parameter descriptions and valid ranges
//! - Intuitive editing controls with visual feedback
//! - Educational context for each parameter's purpose

use crate::{
    config::{Activation, Config, Engine, FitnessFunc, MutationStrategy, ReproductionStrategy},
    tui::app::{App, AppState},
    Population,
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Table},
    Frame,
};
use std::{sync::mpsc, thread};

/// Parameter metadata for enhanced UI display and validation.
struct ParameterInfo {
    name: &'static str,
    description: &'static str,
    value_type: &'static str,
    valid_range: &'static str,
    educational_note: &'static str,
}

/// Comprehensive parameter information for the configuration UI.
const PARAMETER_INFO: &[ParameterInfo] = &[
    ParameterInfo {
        name: "Engine",
        description: "Neural network computation backend",
        value_type: "Engine",
        valid_range: "CPU, GPU",
        educational_note: "CPU is reliable and fast for small networks. GPU can accelerate large populations but has setup overhead.",
    },
    ParameterInfo {
        name: "Generations",
        description: "Number of evolutionary iterations to run",
        value_type: "Number", 
        valid_range: "10-10000",
        educational_note: "More generations allow better evolution but take longer. 100-500 is typically sufficient for Pong.",
    },
    ParameterInfo {
        name: "Population Size",
        description: "Number of individuals in each generation",
        value_type: "Number",
        valid_range: "10-1000",
        educational_note: "Larger populations explore more solutions but require more computation. 50-200 works well for most problems.",
    },
    ParameterInfo {
        name: "Elite Count",
        description: "Top individuals preserved unchanged each generation",
        value_type: "Number",
        valid_range: "1-20% of population",
        educational_note: "Elitism ensures good solutions aren't lost. Too many elites reduce exploration, too few lose progress.",
    },
    ParameterInfo {
        name: "Mutation Rate", 
        description: "Probability each gene will be mutated",
        value_type: "Probability",
        valid_range: "0.00-1.00",
        educational_note: "Higher rates increase exploration but may disrupt good solutions. 0.01-0.1 is typical for neural networks.",
    },
    ParameterInfo {
        name: "Mutation Strength",
        description: "Maximum magnitude of mutations",
        value_type: "Number",
        valid_range: "0.01-2.00",
        educational_note: "Controls how large mutations can be. Smaller values make fine adjustments, larger values enable big changes.",
    },
    ParameterInfo {
        name: "Activation",
        description: "Non-linear function applied in hidden layers",
        value_type: "Function",
        valid_range: "Tanh, ReLU, Sigmoid, etc.",
        educational_note: "Different functions have different learning properties. Tanh is zero-centered, ReLU avoids vanishing gradients.",
    },
    ParameterInfo {
        name: "Reproduction",
        description: "How parent selection and crossover work",
        value_type: "Strategy",
        valid_range: "Tournament, Roulette, etc.",
        educational_note: "Tournament selection is robust and maintains selection pressure. Roulette is simpler but can lose diversity.",
    },
    ParameterInfo {
        name: "Mutation Strategy",
        description: "Approach to introducing genetic variation",
        value_type: "Strategy", 
        valid_range: "Modern, CppEquivalent",
        educational_note: "Modern allows multiple mutations per individual. CppEquivalent mutates exactly one gene for comparison.",
    },
    ParameterInfo {
        name: "Fitness Function",
        description: "How to evaluate individual performance",
        value_type: "Function",
        valid_range: "CppEquivalent, WinLoss",
        educational_note: "CppEquivalent rewards ball returns and shots. WinLoss only cares about winning games.",
    },
    ParameterInfo {
        name: "Random Ball Direction",
        description: "Whether ball direction varies each game",
        value_type: "Boolean",
        valid_range: "true, false",
        educational_note: "Randomization makes training more robust but evaluation less consistent. Enable for diverse training.",
    },
    ParameterInfo {
        name: "Concurrent Training",
        description: "Whether to use parallel processing",
        value_type: "Boolean",
        valid_range: "true, false", 
        educational_note: "Parallel processing speeds up training on multi-core systems but adds complexity and memory usage.",
    },
];

/// Creates a vector of configuration items for display with validation.
fn get_config_items(config: &Config) -> Vec<(&'static str, String, bool)> {
    let mut items = vec![
        ("Engine", config.engine.to_string(), true),
        ("Generations", config.generations.to_string(), config.generations >= 10),
        ("Population Size", config.population_size.to_string(), config.population_size >= 10),
        ("Elite Count", config.elite_count.to_string(), 
         config.elite_count >= 1 && config.elite_count <= config.population_size / 5),
    ];

    // Only show mutation parameters when using Modern mutation strategy
    if matches!(config.mutation_strategy, MutationStrategy::Modern) {
        items.push(("Mutation Rate", format!("{:.3}", config.mutation_rate), 
                   config.mutation_rate >= 0.0 && config.mutation_rate <= 1.0));
        items.push(("Mutation Strength", format!("{:.3}", config.mutation_strength),
                   config.mutation_strength >= 0.01 && config.mutation_strength <= 2.0));
    }

    items.extend([
        ("Activation", config.activation.to_string(), true),
        ("Reproduction", config.reproduction_strategy.to_string(), true),
        ("Mutation Strategy", config.mutation_strategy.to_string(), true),
        ("Fitness Function", config.fitness_func.to_string(), true),
        ("Random Ball Direction", config.random_ball_direction.to_string(), true),
        ("Concurrent Training", config.concurrent.to_string(), true),
    ]);

    items
}

/// Validates the entire configuration and returns issues if any.
fn validate_config(config: &Config) -> Vec<String> {
    let mut issues = Vec::new();
    
    if config.generations < 10 {
        issues.push("Generations too low (minimum 10)".to_string());
    }
    if config.generations > 10000 {
        issues.push("Generations very high (consider reducing for faster results)".to_string());
    }
    
    if config.population_size < 10 {
        issues.push("Population size too small (minimum 10)".to_string());
    }
    if config.population_size > 1000 {
        issues.push("Population size very large (will be slow)".to_string());
    }
    
    if config.elite_count >= config.population_size / 2 {
        issues.push("Too many elites (reduce exploration)".to_string());
    }
    
    if matches!(config.mutation_strategy, MutationStrategy::Modern) {
        if config.mutation_rate > 0.5 {
            issues.push("Mutation rate very high (may disrupt evolution)".to_string());
        }
        if config.mutation_strength > 1.0 {
            issues.push("Mutation strength high (may cause instability)".to_string());
        }
    }
    
    issues
}

/// Enhanced configuration UI with comprehensive parameter display and help.
pub fn draw_config_ui(f: &mut Frame, app: &mut App, area: Rect) {
    // Split the area into config table, validation, and help panel
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(12),      // Config table
            Constraint::Length(4),    // Validation status
            Constraint::Length(8),    // Parameter help
        ])
        .split(area);

    // Draw main configuration table
    draw_config_table(f, app, chunks[0]);
    
    // Draw validation panel
    draw_validation_panel(f, app, chunks[1]);
    
    // Draw parameter explanation panel
    draw_enhanced_parameter_explanation(f, app, chunks[2]);
}

/// Draws the configuration table with enhanced styling and validation indicators.
fn draw_config_table(f: &mut Frame, app: &mut App, area: Rect) {
    let items = get_config_items(&app.config);
    let rows: Vec<Row> = items
        .iter()
        .map(|(key, value, is_valid)| {
            let status_symbol = if *is_valid { "✓" } else { "⚠" };
            let _status_color = if *is_valid { Color::Green } else { Color::Yellow };
            let key_color = if *is_valid { Color::Cyan } else { Color::Yellow };
            
            Row::new(vec![
                Cell::from(format!("{} {}", status_symbol, key))
                    .style(Style::default().fg(key_color)),
                Cell::from(value.clone())
                    .style(Style::default().fg(Color::White)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        &[Constraint::Percentage(60), Constraint::Percentage(40)],
    )
    .block(
        Block::default()
            .title("🔧 Training Configuration")
            .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    )
    .header(
        Row::new(vec!["Parameter", "Value"])
            .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
            .bottom_margin(1),
    )
    .row_highlight_style(
        Style::default()
            .bg(Color::Blue)
            .add_modifier(Modifier::BOLD)
    );

    f.render_stateful_widget(table, area, &mut app.config_editor.state);
}

/// Draws a validation panel showing configuration issues and readiness status.
fn draw_validation_panel(f: &mut Frame, app: &App, area: Rect) {
    let issues = validate_config(&app.config);
    let is_ready = issues.is_empty();
    
    let (status_text, status_color) = if is_ready {
        ("✅ Configuration Ready - Press Enter to Start Training", Color::Green)
    } else {
        ("⚠️  Configuration Issues Found", Color::Yellow)
    };
    
    let mut text = vec![
        Line::from(vec![
            Span::styled(status_text, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        ]),
    ];
    
    if !issues.is_empty() {
        text.push(Line::from(""));
        for issue in issues.iter().take(2) { // Show max 2 issues to fit
            text.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::Yellow)),
                Span::styled(issue, Style::default().fg(Color::White)),
            ]));
        }
        if issues.len() > 2 {
            text.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::Yellow)),
                Span::styled(format!("... and {} more", issues.len() - 2), Style::default().fg(Color::Gray)),
            ]));
        }
    }

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title("🎯 Status")
                .title_style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(if is_ready { Color::Green } else { Color::Yellow })),
        )
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
}

/// Enhanced parameter explanation with comprehensive information.
fn draw_enhanced_parameter_explanation(f: &mut Frame, app: &App, area: Rect) {
    let items = get_config_items(&app.config);
    let selected_idx = app.config_editor.state.selected().unwrap_or(0);
    
    if selected_idx >= items.len() || selected_idx >= PARAMETER_INFO.len() {
        return;
    }
    
    let param_info = &PARAMETER_INFO[selected_idx];
    let (_name, current_value, is_valid) = &items[selected_idx];
    
    let text = vec![
        Line::from(vec![
            Span::styled("Description: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(param_info.description, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Current: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(current_value, Style::default().fg(if *is_valid { Color::Green } else { Color::Red })),
            Span::styled(" | Range: ", Style::default().fg(Color::Gray)),
            Span::styled(param_info.valid_range, Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("💡 ", Style::default().fg(Color::Yellow)),
            Span::styled(param_info.educational_note, Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Controls: ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw("←/→ to change, ↑/↓ to navigate, Enter to start, q to back"),
        ]),
    ];

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title(format!("📋 Parameter: {}", param_info.name))
                .title_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Green)),
        )
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
}

/// Enhanced input handling with better parameter modification and validation.
pub fn handle_config_input(app: &mut App, key_code: KeyCode) {
    let num_items = get_config_items(&app.config).len();
    
    match key_code {
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
            app.state = AppState::MainMenu;
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let next_index = (app.config_editor.state.selected().unwrap_or(0) + 1) % num_items;
            app.config_editor.state.select(Some(next_index));
        }
        KeyCode::Up | KeyCode::Char('k') => {
            let prev_index = (app.config_editor.state.selected().unwrap_or(0) + num_items - 1) % num_items;
            app.config_editor.state.select(Some(prev_index));
        }
        KeyCode::Left | KeyCode::Char('h') => {
            change_config_value(app, false);
        }
        KeyCode::Right | KeyCode::Char('l') => {
            change_config_value(app, true);
        }
        KeyCode::Enter => {
            let issues = validate_config(&app.config);
            if !issues.is_empty() {
                app.error_message = Some(format!(
                    "Configuration has issues:\n\n{}\n\nPlease fix these before starting training.",
                    issues.join("\n")
                ));
                return;
            }
            
            start_training(app);
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            // Reset to defaults
            app.config = Config::default();
            app.success_message = Some("Configuration reset to defaults".to_string());
        }
        _ => {}
    }
}

/// Starts the training process with the current configuration.
fn start_training(app: &mut App) {
    // Start Training
    let (tx, rx) = mpsc::channel();
    app.tx = Some(tx.clone());
    app.rx = Some(rx);

    // Initialize training state
    app.training = Some(crate::tui::training::TrainingState::new(&app.config));

    let training_config = app.config.clone();
    let handle = thread::spawn(move || {
        macro_rules! run_evolution_for_engine {
            ($individual_type:ty, $config:expr, $sender:expr) => {{
                let mut pop: Population<$individual_type> = Population::new($config);
                pop.evolve(Some($sender));
            }};
        }
        
        match training_config.engine {
            Engine::Cpu => {
                run_evolution_for_engine!(
                    crate::engines::StackIndividual,
                    training_config,
                    tx.clone()
                )
            }
            Engine::Gpu => {
                run_evolution_for_engine!(
                    crate::engines::GpuIndividual, 
                    training_config, 
                    tx.clone()
                )
            }
        }
        
        // After evolution is done, send the finished signal.
        if tx.send(crate::tui::training::TrainingMessage::Finished).is_err() {
            // The UI has likely been closed, so we don't need to log an error.
        }
    });
    
    app.training_thread = Some(handle);
    app.state = AppState::Training;
    app.success_message = Some("Training started! Watch the evolution progress.".to_string());
}

/// Enhanced parameter modification with proper bounds checking and user feedback.
fn change_config_value(app: &mut App, increase: bool) {
    let config = &mut app.config;
    let items = get_config_items(config);
    
    if let Some(selected) = app.config_editor.state.selected() {
        if selected >= items.len() {
            return;
        }
        
        let item_name = items[selected].0;
        let old_value = items[selected].1.clone();
        
        match item_name {
            "Engine" => {
                config.engine = match config.engine {
                    Engine::Cpu => Engine::Gpu,
                    Engine::Gpu => Engine::Cpu,
                };
            }
            "Generations" => {
                let step = if config.generations < 100 { 10 } else { 50 };
                if increase {
                    config.generations = (config.generations + step).min(10000);
                } else {
                    config.generations = config.generations.saturating_sub(step).max(10);
                }
            }
            "Population Size" => {
                let step = if config.population_size < 50 { 5 } else { 10 };
                if increase {
                    config.population_size = (config.population_size + step).min(1000);
                } else {
                    config.population_size = config.population_size.saturating_sub(step).max(10);
                }
            }
            "Elite Count" => {
                let max_elites = config.population_size / 5; // Maximum 20% elites
                if increase {
                    config.elite_count = (config.elite_count + 1).min(max_elites).max(1);
                } else {
                    config.elite_count = config.elite_count.saturating_sub(1).max(1);
                }
            }
            "Mutation Rate" => {
                let step = 0.005; // Smaller steps for precision
                if increase {
                    config.mutation_rate = (config.mutation_rate + step).min(1.0);
                } else {
                    config.mutation_rate = (config.mutation_rate - step).max(0.0);
                }
            }
            "Mutation Strength" => {
                let step = 0.01;
                if increase {
                    config.mutation_strength = (config.mutation_strength + step).min(2.0);
                } else {
                    config.mutation_strength = (config.mutation_strength - step).max(0.01);
                }
            }
            "Activation" => {
                config.activation = match config.activation {
                    Activation::ClampedLinear => Activation::Tanh,
                    Activation::Tanh => Activation::Relu,
                    Activation::Relu => Activation::Atan,
                    Activation::Atan => Activation::Sigmoid,
                    Activation::Sigmoid => Activation::Linear,
                    Activation::Linear => Activation::ClampedLinear,
                };
            }
            "Reproduction" => {
                config.reproduction_strategy = match config.reproduction_strategy {
                    ReproductionStrategy::CppEquivalent => ReproductionStrategy::EliteCrossover,
                    ReproductionStrategy::EliteCrossover => ReproductionStrategy::CppEquivalent,
                };
            }
            "Mutation Strategy" => {
                config.mutation_strategy = match config.mutation_strategy {
                    MutationStrategy::Modern => MutationStrategy::CppEquivalent,
                    MutationStrategy::CppEquivalent => MutationStrategy::Modern,
                };
            }
            "Fitness Function" => {
                config.fitness_func = match config.fitness_func {
                    FitnessFunc::CppEquivalent => FitnessFunc::ReturnFocused,
                    FitnessFunc::ReturnFocused => FitnessFunc::VictoryOptimized,
                    FitnessFunc::VictoryOptimized => FitnessFunc::CppEquivalent,
                };
            }
            "Random Ball Direction" => {
                config.random_ball_direction = !config.random_ball_direction;
            }
            "Concurrent Training" => {
                config.concurrent = !config.concurrent;
            }
            _ => {}
        }
        
        // Provide user feedback for significant changes
        let new_value = get_config_items(config)[selected].1.clone();
        if new_value != old_value {
            app.success_message = Some(format!("Changed {} from {} to {}", item_name, old_value, new_value));
        }
    }
}
