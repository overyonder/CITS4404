//! The training configuration screen component.

use crate::{
    config::{Activation, Config, Engine, FitnessFunc, MutationStrategy, ReproductionStrategy},
    tui::app::{App, AppState},
    Population,
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType, Borders, Cell, Row, Table},
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
        (
            "Reproduction",
            config.reproduction_strategy.to_string(),
        ),
        ("Mutation", config.mutation_strategy.to_string()),
        ("Fitness", config.fitness_func.to_string()),
        ("Concurrent", config.concurrent.to_string()),
    ]
}

/// Draws the UI for configuring a new training session.
pub fn draw_config_ui(f: &mut Frame, app: &mut App, area: Rect) {
    let items = get_config_items(&app.config);
    let rows: Vec<Row> = items
        .iter()
        .map(|(key, value)| {
            Row::new(vec![
                Cell::from(*key).style(Style::default().fg(Color::Cyan)),
                Cell::from(value.clone()),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        &[Constraint::Percentage(50), Constraint::Percentage(50)],
    )
    .block(
        Block::default()
            .title("Training Configuration (Use Up/Down to navigate, Left/Right to change, Enter to start, 'q' to go back)")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    )
    .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_stateful_widget(table, area, &mut app.config_editor.state);
}

/// Handles user input on the configuration screen.
pub fn handle_config_input(app: &mut App, key_code: KeyCode) {
    let num_items = get_config_items(&app.config).len();
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.state = AppState::MainMenu;
        }
        KeyCode::Down => {
            let next_index = (app.config_editor.state.selected().unwrap_or(0) + 1) % num_items;
            app.config_editor.state.select(Some(next_index));
        }
        KeyCode::Up => {
            let prev_index = (app.config_editor.state.selected().unwrap_or(0) + num_items - 1)
                % num_items;
            app.config_editor.state.select(Some(prev_index));
        }
        KeyCode::Left => {
            change_config_value(app, false);
        }
        KeyCode::Right => {
            change_config_value(app, true);
        }
        KeyCode::Enter => {
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
                    Engine::Stack => {
                        run_evolution_for_engine!(
                            crate::engines::StackIndividual,
                            training_config,
                            tx.clone()
                        )
                    }
                    Engine::Heap => {
                        run_evolution_for_engine!(
                            crate::engines::HeapIndividual,
                            training_config,
                            tx.clone()
                        )
                    }
                    Engine::Simd => {
                        run_evolution_for_engine!(
                            crate::engines::SimdIndividual,
                            training_config,
                            tx.clone()
                        )
                    }
                    Engine::Gpu => {
                        run_evolution_for_engine!(crate::engines::GpuIndividual, training_config, tx.clone())
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
        _ => {}
    }
}

/// Helper function to modify a configuration value based on the selected index.
fn change_config_value(app: &mut App, increase: bool) {
    let config = &mut app.config;
    match app.config_editor.state.selected().unwrap_or(0) {
        0 => {
            // Engine
            config.engine = match config.engine {
                Engine::Stack => if increase { Engine::Heap } else { Engine::Gpu },
                Engine::Heap => if increase { Engine::Simd } else { Engine::Stack },
                Engine::Simd => if increase { Engine::Gpu } else { Engine::Heap },
                Engine::Gpu => if increase { Engine::Stack } else { Engine::Simd },
            };
        }
        1 => {
            // Generations
            let step = 50;
            if increase {
                config.generations += step;
            } else {
                config.generations = config.generations.saturating_sub(step).max(10);
            }
        }
        2 => {
            // Population Size
            let step = 10;
            if increase {
                config.population_size += step;
            } else {
                config.population_size = config.population_size.saturating_sub(step).max(10);
            }
        }
        3 => {
            // Elite Count
            if increase {
                config.elite_count += 1;
            } else {
                config.elite_count = config.elite_count.saturating_sub(1).max(1);
            }
        }
        4 => {
            // Mutation Rate
            let step = 0.01;
            if increase {
                config.mutation_rate = (config.mutation_rate + step).min(1.0);
            } else {
                config.mutation_rate = (config.mutation_rate - step).max(0.0);
            }
        }
        5 => {
            // Mutation Strength
            let step = 0.05;
            if increase {
                config.mutation_strength = (config.mutation_strength + step).min(2.0);
            } else {
                config.mutation_strength = (config.mutation_strength - step).max(0.0);
            }
        }
        6 => {
            // Activation
            config.activation = match config.activation {
                Activation::ClampedLinear => if increase { Activation::Tanh } else { Activation::Linear },
                Activation::Tanh => if increase { Activation::Relu } else { Activation::ClampedLinear },
                Activation::Relu => if increase { Activation::Atan } else { Activation::Tanh },
                Activation::Atan => if increase { Activation::Sigmoid } else { Activation::Relu },
                Activation::Sigmoid => if increase { Activation::Linear } else { Activation::Atan },
                Activation::Linear => if increase { Activation::ClampedLinear } else { Activation::Sigmoid },
            };
        }
        7 => {
            // Reproduction Strategy
            config.reproduction_strategy = match config.reproduction_strategy {
                ReproductionStrategy::CppEquivalent => ReproductionStrategy::Modern,
                ReproductionStrategy::Modern => ReproductionStrategy::CppEquivalent,
            }
        }
        8 => {
            // Mutation Strategy
            config.mutation_strategy = match config.mutation_strategy {
                MutationStrategy::CppEquivalent => MutationStrategy::Modern,
                MutationStrategy::Modern => MutationStrategy::CppEquivalent,
            }
        }
        9 => {
            // Fitness Function
            config.fitness_func = match config.fitness_func {
                FitnessFunc::CppEquivalent => if increase { FitnessFunc::Balanced } else { FitnessFunc::Performance },
                FitnessFunc::Balanced => if increase { FitnessFunc::Performance } else { FitnessFunc::CppEquivalent },
                FitnessFunc::Performance => if increase { FitnessFunc::CppEquivalent } else { FitnessFunc::Balanced },
            }
        }
        10 => {
            // Concurrent
            config.concurrent = !config.concurrent;
        }
        _ => {}
    }
}
