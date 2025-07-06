//! The configuration screen component.

use crate::{
    config::{Activation, Config, Engine, FitnessFunc, ReproductionStrategy},
    engines::{GpuIndividual, HeapIndividual, SimdIndividual, StackIndividual},
    population::Population,
    tui::{
        app::{App, AppState},
        training::TrainingState,
    },
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph},
    Frame,
};
use std::{sync::mpsc, thread};

/// Creates a vector of configuration items for display.
fn get_config_items(config: &Config) -> Vec<(&'static str, String)> {
    vec![
        ("Engine", config.engine.to_string()),
        ("Generations", config.generations.to_string()),
        ("Population Size", config.population_size.to_string()),
        ("Elite Count", config.elite_count.to_string()),
        ("Mutation Rate", format!("{:.2}", config.mutation_rate)),
        (
            "Mutation Strength",
            format!("{:.2}", config.mutation_strength),
        ),
        ("Activation", config.activation.to_string()),
        ("Concurrent", config.concurrent.to_string()),
        ("Fitness Func", config.fitness_func.to_string()),
        (
            "Reproduction",
            config.reproduction_strategy.to_string(),
        ),
    ]
}

/// Draws the UI for the configuration editor.
pub fn draw_config_ui(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Min(0),    // List
            Constraint::Length(3), // Footer
        ])
        .split(area);

    let title = Paragraph::new("Configuration")
        .block(Block::default().borders(Borders::TOP | Borders::LEFT | Borders::RIGHT));
    f.render_widget(title, chunks[0]);

    let config_items = get_config_items(&app.config);
    let items: Vec<ListItem> = config_items
        .iter()
        .enumerate()
        .map(|(i, (name, value))| {
            let style = if i == app.config_editor.selected_index {
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .bg(Color::Blue)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:<20}", name), style),
                Span::raw(" | "),
                Span::styled(value.to_string(), style),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::LEFT | Borders::RIGHT)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(list, chunks[1]);

    let footer =
        Paragraph::new("Use Up/Down to navigate, Left/Right to change, Enter to Start, Esc to go back.")
            .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);
}

/// Handles user input on the configuration screen.
pub fn handle_config_input(app: &mut App, key_code: KeyCode) {
    let num_items = get_config_items(&app.config).len();
    match key_code {
        KeyCode::Up => {
            app.config_editor.selected_index = app.config_editor.selected_index.saturating_sub(1);
        }
        KeyCode::Down => {
            if app.config_editor.selected_index < num_items - 1 {
                app.config_editor.selected_index += 1;
            }
        }
        KeyCode::Left => change_config_value(app, false),
        KeyCode::Right => change_config_value(app, true),
        KeyCode::Enter => start_training(app),
        KeyCode::Esc | KeyCode::Char('q') => {
            app.state = AppState::MainMenu;
        }
        _ => {}
    }
}

/// Starts the training process in a background thread.
fn start_training(app: &mut App) {
    // Create a channel for the training thread to send updates to the UI thread.
    let (tx, rx) = mpsc::channel();
    app.tx = Some(tx);
    app.rx = Some(rx);

    // Initialize the training state.
    app.training = Some(TrainingState::new(&app.config));
    app.state = AppState::Training;

    // Clone the necessary data to move into the training thread.
    let config = app.config.clone();
    // Clone the Option<Sender>. This is cheap, avoids unwrapping, and is safer.
    let tx_option = app.tx.clone();

    // Spawn the training thread.
    thread::spawn(move || {
        // This macro reduces code duplication for running evolution with different
        // individual types (engines).
        macro_rules! run_evolution {
            ($individual_type:ty, $config:expr) => {{
                let mut pop: Population<$individual_type> = Population::new($config.clone());
                // The `evolve` function takes an `Option<Sender<TrainingMessage>>`.
                // We clone the Option for each potential call within the macro expansion.
                let _best_individual = pop.evolve(tx_option.clone());
            }};
        }

        // Dispatch to the correct engine type.
        match config.engine {
            Engine::Stack => run_evolution!(StackIndividual, config),
            Engine::Heap => run_evolution!(HeapIndividual, config),
            Engine::Simd => run_evolution!(SimdIndividual, config),
            Engine::Gpu => run_evolution!(GpuIndividual, config),
        };
    });
}

/// Helper function to modify a configuration value based on the selected index.
fn change_config_value(app: &mut App, increase: bool) {
    let config = &mut app.config;
    match app.config_editor.selected_index {
        0 => {
            // Engine
            let current_engine_index = match config.engine {
                Engine::Stack => 0,
                Engine::Heap => 1,
                Engine::Simd => 2,
                Engine::Gpu => 3,
            };
            let next_index = if increase {
                (current_engine_index + 1) % 4
            } else {
                (current_engine_index + 3) % 4
            };
            config.engine = match next_index {
                0 => Engine::Stack,
                1 => Engine::Heap,
                2 => Engine::Simd,
                _ => Engine::Gpu,
            };
        }
        1 => {
            // Generations
            let step = 10;
            if increase {
                config.generations += step;
            } else {
                config.generations = config.generations.saturating_sub(step);
            }
        }
        2 => {
            // Population Size
            let step = 10;
            if increase {
                config.population_size += step;
            } else {
                config.population_size = config.population_size.saturating_sub(step);
            }
        }
        3 => {
            // Elite Count
            if increase {
                config.elite_count += 1;
            } else {
                config.elite_count = config.elite_count.saturating_sub(1);
            }
        }
        4 => {
            // Mutation Rate
            let step = 0.01;
            if increase {
                config.mutation_rate += step;
            } else {
                config.mutation_rate = (config.mutation_rate - step).max(0.0);
            }
        }
        5 => {
            // Mutation Strength
            let step = 0.01;
            if increase {
                config.mutation_strength += step;
            } else {
                config.mutation_strength = (config.mutation_strength - step).max(0.0);
            }
        }
        6 => {
            // Activation
            let current_activation_index = match config.activation {
                Activation::Tanh => 0,
                Activation::Relu => 1,
                Activation::Atan => 2,
                Activation::Linear => 3,
                Activation::Sigmoid => 4,
            };
            let next_index = if increase {
                (current_activation_index + 1) % 5
            } else {
                (current_activation_index + 4) % 5
            };
            config.activation = match next_index {
                0 => Activation::Tanh,
                1 => Activation::Relu,
                2 => Activation::Atan,
                3 => Activation::Linear,
                4 => Activation::Sigmoid,
                _ => Activation::default(),
            };
        }
        7 => {
            // Concurrent
            config.concurrent = !config.concurrent;
        }
        8 => {
            // Fitness Function
            let current_fitness_index = match config.fitness_func {
                FitnessFunc::CppEquivalent => 0,
                FitnessFunc::Balanced => 1,
                FitnessFunc::Performance => 2,
            };
            let next_index = if increase {
                (current_fitness_index + 1) % 3
            } else {
                (current_fitness_index + 2) % 3
            };
            config.fitness_func = match next_index {
                0 => FitnessFunc::CppEquivalent,
                1 => FitnessFunc::Balanced,
                _ => FitnessFunc::Performance,
            };
        }
        9 => {
            // Reproduction Strategy
            config.reproduction_strategy = match config.reproduction_strategy {
                ReproductionStrategy::CppEquivalent => ReproductionStrategy::Modern,
                ReproductionStrategy::Modern => ReproductionStrategy::CppEquivalent,
            };
        }
        _ => {}
    }
}
