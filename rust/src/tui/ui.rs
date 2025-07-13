//! UI rendering and main event loop for the TUI.

use crate::{
    tui::{
        app::{App, AppState},
        components,
        training::TrainingMessage,
    },
    Individual,
};
use crossterm::event::{self, Event, KeyEventKind};
use ratatui::{
    backend::CrosstermBackend,
    style::Color,
    Frame, Terminal,
};
use std::{
    fs,
    io::{Result, Stdout},
    path::Path,
};

/// Main event loop for the TUI, handling events and rendering.
pub fn main_event_loop(
    app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
) -> Result<()> {
    while app.state != AppState::Exiting {
        terminal.draw(|f| ui_builder(f, app))?;
        handle_events(app)?;
    }
    Ok(())
}

/// Handles state updates, user input, and messages from background threads.
fn handle_events(app: &mut App) -> Result<()> {
    // Handle non-blocking state updates (e.g., background threads, simulation steps).
    match app.state {
        AppState::Training => {
            // Drain the message queue on each tick to prevent the UI from lagging.
            while let Ok(msg) = app.rx.as_ref().unwrap().try_recv() {
                if let Some(ts) = app.training.as_mut() {
                    match msg {
                        TrainingMessage::Progress {
                            generation,
                            best_fitness,
                            genome_weights,
                            total_matches_simulated,
                            training_rate,
                            improvement_rate,
                        } => {
                            ts.current_generation = generation;
                            ts.fitness_history.push(best_fitness);
                            ts.total_matches_simulated = total_matches_simulated;
                            ts.training_rate_history.push(training_rate);
                            
                            // Use the new sampling method for improvement rate averaging
                            ts.add_improvement_rate_sample(improvement_rate);
                            
                            // Only update champion genome if this is a new all-time best
                            if best_fitness > ts.best_fitness {
                                ts.best_fitness = best_fitness;
                                ts.genome_weights = genome_weights.clone();
                                app.best_genome = Some(genome_weights);
                            }
                        }
                        TrainingMessage::Finished => {
                            ts.running = false;
                            // Save the best genome when training is finished.
                            if let Some(genome) = &app.best_genome {
                                if let Ok(weights_array) = genome.clone().try_into() {
                                    let temp_individual = crate::engines::StackIndividual {
                                        weights: weights_array,
                                    };

                                    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
                                    let filename = format!("models/{}_champion.json", timestamp);

                                    // Clone the config, and update with actual training results
                                    let mut config_to_save = app.config.clone();
                                    config_to_save.name = Some(
                                        Path::new(&filename)
                                            .file_stem()
                                            .unwrap()
                                            .to_string_lossy()
                                            .to_string(),
                                    );
                                    // Update with actual generations completed (normal completion)
                                    config_to_save.generations = ts.current_generation as u32;

                                    if let Err(e) = temp_individual.save(&filename, &config_to_save)
                                    {
                                        app.error_message =
                                            Some(format!("Failed to save champion: {}", e));
                                    } else {
                                        app.success_message =
                                            Some(format!("Champion saved to {}", filename));
                                    }
                                } else {
                                    app.error_message = Some(format!(
                                        "Genome size mismatch. Expected {}, found {}.",
                                        crate::engines::constants::TOTAL_WEIGHTS,
                                        genome.len()
                                    ));
                                }
                            }
                        }
                        TrainingMessage::EarlyStopping { final_generation, best_fitness } => {
                            ts.running = false;
                            ts.current_generation = final_generation;
                            if best_fitness > ts.best_fitness {
                                ts.best_fitness = best_fitness;
                            }
                            
                            // Note: Champion genome is automatically saved during progress updates
                            // when a new best fitness is achieved, so no additional saving needed here.
                            app.success_message = Some(format!(
                                "Training stopped early at generation {} (fitness: {:.2})", 
                                final_generation, best_fitness
                            ));
                        }
                    }
                }
            }
        }
        AppState::Simulation => {
            if let Some(sim) = app.simulation.as_mut() {
                sim.step(&app.config);
            }
        }
        _ => {}
    }

    // Handle user input with a timeout to allow for smooth animation.
    if crossterm::event::poll(std::time::Duration::from_millis(100))? {
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                // If an error message is visible, the first key press clears it
                // and does nothing else. This makes the error popup modal.
                if app.error_message.is_some() {
                    app.error_message = None;
                    return Ok(()); // Consume the key press and do nothing else.
                }
                if app.success_message.is_some() {
                    app.success_message = None;
                    return Ok(());
                }

                match app.state {
                    AppState::MainMenu => {
                        components::main_menu::handle_main_menu_input(app, key.code)
                    }
                    AppState::SimulationSetup => {
                        components::simulation_setup_ui::handle_simulation_setup_input(
                            app, key.code,
                        )
                    }
                    AppState::Configuring => {
                        components::config_ui::handle_config_input(app, key.code)
                    }
                    AppState::Training => {
                        components::training_ui::handle_training_input(app, key.code)
                    }
                    AppState::Simulation => {
                        components::simulation_ui::handle_simulation_input(app, key.code)
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// Builds the UI for the current application state.
fn ui_builder(f: &mut Frame, app: &mut App) {
    match app.state {
        AppState::MainMenu => components::main_menu::draw_main_menu(f, app, f.area()),
        AppState::Training => components::training_ui::draw_training_ui(f, app, f.area()),
        AppState::SimulationSetup => {
            components::simulation_setup_ui::draw_simulation_setup_ui(f, app, f.area())
        }
        AppState::Simulation => components::simulation_ui::draw_simulation_ui(f, app, f.area()),
        AppState::Configuring => components::config_ui::draw_config_ui(f, app, f.area()),
        AppState::Exiting => {} // Do nothing, the app will exit
    };

    if let Some(msg) = &app.error_message {
        components::message_popup::draw_message_popup(f, msg, "Error", Color::Red, f.area());
    } else if let Some(msg) = &app.success_message {
        components::message_popup::draw_message_popup(f, msg, "Success", Color::Green, f.area());
    }
}

/// Entrypoint for the TUI from main.rs
pub fn run_app() -> Result<()> {
    // Create the models directory if it doesn't exist.
    fs::create_dir_all("models")?;

    let mut terminal = setup_terminal()?;
    let mut app = App::new();
    // The main_event_loop will run until the user exits.
    let res = main_event_loop(&mut app, &mut terminal);
    restore_terminal()?;
    res
}

fn setup_terminal() -> std::io::Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    use crossterm::execute;
    use crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};
    use std::io::stdout;
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    Terminal::new(backend)
}

fn restore_terminal() -> std::io::Result<()> {
    use crossterm::execute;
    use crossterm::terminal::disable_raw_mode;
    use crossterm::terminal::LeaveAlternateScreen;
    use std::io::stdout;
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}
