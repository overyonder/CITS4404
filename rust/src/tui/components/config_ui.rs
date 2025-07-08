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
    valid_range: &'static str,
    educational_note: &'static str,
}

/// Comprehensive parameter information for the configuration UI.
const PARAMETER_INFO: &[ParameterInfo] = &[
    ParameterInfo {
        name: "Engine",
        description: "Neural network computation backend",
        valid_range: "CPU, GPU",
        educational_note: "CPU is reliable and fast for small networks. GPU can accelerate large populations but has setup overhead.",
    },
    ParameterInfo {
        name: "Generations",
        description: "Number of evolutionary iterations to run",
        valid_range: "10-10000",
        educational_note: "More generations allow better evolution but take longer. 100-500 is typically sufficient for Pong.",
    },
    ParameterInfo {
        name: "Population Size",
        description: "Number of individuals in each generation",
        valid_range: "10-8000",
        educational_note: "Larger populations explore more solutions but require more computation. 50-200 works well for most problems.",
    },
    ParameterInfo {
        name: "Elite Count",
        description: "Top individuals preserved unchanged each generation",
        valid_range: "1-20% of population",
        educational_note: "Elitism ensures good solutions aren't lost. Too many elites reduce exploration, too few lose progress.",
    },
    ParameterInfo {
        name: "Mutation Rate", 
        description: "Probability each gene will be mutated",
        valid_range: "0.00-1.00",
        educational_note: "Higher rates increase exploration but may disrupt good solutions. 0.01-0.1 is typical for neural networks.",
    },
    ParameterInfo {
        name: "Mutation Strength",
        description: "Maximum magnitude of mutations",
        valid_range: "0.01-2.00",
        educational_note: "Controls how large mutations can be. Smaller values make fine adjustments, larger values enable big changes.",
    },
    ParameterInfo {
        name: "Activation",
        description: "Non-linear function applied in hidden layers",
        valid_range: "Tanh, ReLU, Sigmoid, etc.",
        educational_note: "Different functions have different learning properties. Tanh is zero-centered, ReLU avoids vanishing gradients.",
    },
    ParameterInfo {
        name: "Reproduction",
        description: "How parent selection and crossover work",
        valid_range: "Tournament, Roulette, etc.",
        educational_note: "Tournament selection is robust and maintains selection pressure. Roulette is simpler but can lose diversity.",
    },
    ParameterInfo {
        name: "Mutation Strategy",
        description: "Approach to introducing genetic variation",
        valid_range: "Modern, CppEquivalent",
        educational_note: "Modern allows multiple mutations per individual. CppEquivalent mutates exactly one gene for comparison.",
    },
    ParameterInfo {
        name: "Fitness Function",
        description: "How to evaluate individual performance",
        valid_range: "CppEquivalent, WinLoss",
        educational_note: "CppEquivalent rewards ball returns and shots. WinLoss only cares about winning games.",
    },
    ParameterInfo {
        name: "Early Stopping Patience",
        description: "Stop if no improvement for this many generations",
        valid_range: "None, 10-500",
        educational_note: "Prevents wasted computation when evolution has plateaued. None disables early stopping. 50-200 is typical for most problems.",
    },
    ParameterInfo {
        name: "Fitness Threshold",
        description: "Minimum improvement required to continue training",
        valid_range: "None, 0.001-1.0",
        educational_note: "Defines 'meaningful' progress to avoid training on tiny improvements. None disables threshold checking. 0.01-0.1 is typical.",
    },
    ParameterInfo {
        name: "Track Diversity",
        description: "Monitor population genetic diversity during training",
        valid_range: "true, false",
        educational_note: "Helps diagnose premature convergence issues but adds computational overhead. Enable for research/debugging.",
    },
    ParameterInfo {
        name: "Random Ball Direction",
        description: "Whether ball direction varies each game",
        valid_range: "true, false",
        educational_note: "Randomization makes training more robust but evaluation less consistent. Enable for diverse training.",
    },
    ParameterInfo {
        name: "Concurrent Training",
        description: "Whether to use parallel processing",
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
        // Early stopping and convergence parameters
        ("Early Stopping Patience", 
         config.early_stopping_patience.map_or("None".to_string(), |p| p.to_string()), 
         true),
        ("Fitness Threshold", 
         config.fitness_threshold.map_or("None".to_string(), |t| format!("{:.4}", t)), 
         true),
        ("Track Diversity", config.track_diversity.to_string(), true),
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
    if config.population_size > 8000 {
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
                Span::styled("• ", Style::default().fg(Color::Red)),
                Span::styled(issue, Style::default().fg(Color::White)),
            ]));
        }
        if issues.len() > 2 {
            text.push(Line::from(vec![
                Span::styled(format!("... and {} more issues", issues.len() - 2), 
                           Style::default().fg(Color::Gray)),
            ]));
        }
    } else {
        text.push(Line::from(""));
        text.push(Line::from(vec![
            Span::styled("Controls: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("↑/↓", Style::default().fg(Color::Yellow)),
            Span::styled(" navigate, ", Style::default().fg(Color::Gray)),
            Span::styled("←/→", Style::default().fg(Color::Yellow)),
            Span::styled(" change, ", Style::default().fg(Color::Gray)),
            Span::styled("'d'", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(" toggle None/Some", Style::default().fg(Color::Gray)),
        ]));
    }

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title("🚦 Status")
                .title_style(Style::default().fg(status_color).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
}

/// Enhanced parameter explanation with comprehensive information.
fn draw_enhanced_parameter_explanation(f: &mut Frame, app: &App, area: Rect) {
    let items = get_config_items(&app.config);
    let selected_idx = app.config_editor.state.selected().unwrap_or(0);
    
    if selected_idx >= items.len() {
        return;
    }
    
    let (param_name, current_value, is_valid) = &items[selected_idx];
    
    // Find matching parameter info
    let param_info = PARAMETER_INFO.iter()
        .find(|p| p.name == *param_name)
        .unwrap_or(&PARAMETER_INFO[0]);

    let status_color = if *is_valid { Color::Green } else { Color::Yellow };
    let border_color = if *is_valid { Color::Cyan } else { Color::Yellow };
    
    // Create enhanced help text with better toggle instructions
    let mut help_lines = vec![
        Line::from(vec![
            Span::styled("Parameter: ", Style::default().fg(Color::Gray)),
            Span::styled(*param_name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Current: ", Style::default().fg(Color::Gray)),
            Span::styled(current_value.clone(), Style::default().fg(status_color)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Description: ", Style::default().fg(Color::Yellow)),
            Span::raw(param_info.description),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Valid Range: ", Style::default().fg(Color::Green)),
            Span::raw(param_info.valid_range),
        ]),
    ];
    
    // Add special instructions for optional parameters that can be toggled with 'd'
    if param_name == &"Early Stopping Patience" || param_name == &"Fitness Threshold" {
        help_lines.push(Line::from(""));
        help_lines.push(Line::from(vec![
            Span::styled("Press 'd' to toggle between ", Style::default().fg(Color::Cyan)),
            Span::styled("None", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(" and ", Style::default().fg(Color::Cyan)),
            Span::styled("Some(default)", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]));
    }
    
    help_lines.push(Line::from(""));
    help_lines.push(Line::from(vec![
        Span::styled("Note: ", Style::default().fg(Color::Blue)),
        Span::raw(param_info.educational_note),
    ]));

    let paragraph = Paragraph::new(help_lines)
        .block(
            Block::default()
                .title("📖 Parameter Guide")
                .title_style(Style::default().fg(border_color).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(border_color)),
        )
        .wrap(ratatui::widgets::Wrap { trim: true });

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
        }
        KeyCode::Char('d') | KeyCode::Char('D') => {
            // Toggle optional parameters between None and Some(default)
            toggle_optional_parameter(app);
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
                // Use GpuIndividual for proper GPU processing
                run_evolution_for_engine!(
                    crate::engines::GpuIndividual<'static>, 
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
        let _old_value = items[selected].1.clone(); // Used for potential debugging
        
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
                    config.population_size = (config.population_size + step).min(8000);
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
                config.activation = if increase {
                    // Forward direction (right arrow)
                    match config.activation {
                        Activation::ClampedLinear => Activation::Tanh,
                        Activation::Tanh => Activation::Relu,
                        Activation::Relu => Activation::Atan,
                        Activation::Atan => Activation::Sigmoid,
                        Activation::Sigmoid => Activation::Linear,
                        Activation::Linear => Activation::ClampedLinear,
                    }
                } else {
                    // Reverse direction (left arrow)
                    match config.activation {
                        Activation::ClampedLinear => Activation::Linear,
                        Activation::Linear => Activation::Sigmoid,
                        Activation::Sigmoid => Activation::Atan,
                        Activation::Atan => Activation::Relu,
                        Activation::Relu => Activation::Tanh,
                        Activation::Tanh => Activation::ClampedLinear,
                    }
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
            "Early Stopping Patience" => {
                let step = 10; // Larger steps for better range
                if increase {
                    config.early_stopping_patience = Some((config.early_stopping_patience.unwrap_or(50) + step).min(500));
                } else {
                    let current = config.early_stopping_patience.unwrap_or(50);
                    if current > step {
                        config.early_stopping_patience = Some((current - step).max(10));
                    }
                }
            }
            "Fitness Threshold" => {
                let step = 0.01; // Larger steps for better control
                if increase {
                    config.fitness_threshold = Some((config.fitness_threshold.unwrap_or(0.01) + step).min(1.0));
                } else {
                    let current = config.fitness_threshold.unwrap_or(0.01);
                    if current > step {
                        config.fitness_threshold = Some((current - step).max(0.001));
                    }
                }
            }
            "Track Diversity" => {
                config.track_diversity = !config.track_diversity;
            }
            "Random Ball Direction" => {
                config.random_ball_direction = !config.random_ball_direction;
            }
            "Concurrent Training" => {
                config.concurrent = !config.concurrent;
            }
            _ => {}
        }
        
        // Note: User feedback removed as requested - no popup on parameter changes
    }
}

/// Toggles optional parameters between None and Some(default_value).
fn toggle_optional_parameter(app: &mut App) {
    let config = &mut app.config;
    let items = get_config_items(config);
    
    if let Some(selected) = app.config_editor.state.selected() {
        if selected >= items.len() {
            return;
        }
        
        let item_name = items[selected].0;
        
        match item_name {
            "Early Stopping Patience" => {
                config.early_stopping_patience = if config.early_stopping_patience.is_some() {
                    None
                } else {
                    Some(50) // Updated default value for better range
                };
            }
            "Fitness Threshold" => {
                config.fitness_threshold = if config.fitness_threshold.is_some() {
                    None
                } else {
                    Some(0.01) // Updated default value for better range
                };
            }
            _ => {
                // For non-optional parameters, do nothing
            }
        }
    }
}
