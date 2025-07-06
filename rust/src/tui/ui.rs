//! UI rendering and main event loop for the TUI.

use crate::{
    cpp_compat,
    tui::{
        app::{App, AppState},
        simulation::SimulationState,
        training::{MatchupState, TrainingMessage},
    },
};
use crossterm::event::{self, Event, KeyEventKind};
use ratatui::{backend::CrosstermBackend, layout::Alignment, widgets::Paragraph, Frame, Terminal};
use std::{
    fs,
    io::{Result, Stdout},
};

// This module is declared in `tui/mod.rs`, so it's available to its sibling `ui.rs`.
use super::components;

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
            if let Ok(msg) = app.rx.as_ref().unwrap().try_recv() {
                if let Some(ts) = app.training.as_mut() {
                    match msg {
                        TrainingMessage::GenerationStart { total_matchups } => {
                            ts.matchups = vec![MatchupState::Pending; total_matchups];
                        }
                        TrainingMessage::MatchupUpdate {
                            matchup_index,
                            state,
                        } => {
                            if let Some(matchup) = ts.matchups.get_mut(matchup_index) {
                                *matchup = state;
                            }
                        }
                        TrainingMessage::Progress {
                            generation,
                            best_fitness,
                            genome_weights,
                            ..
                        } => {
                            ts.current_generation = generation;
                            ts.best_fitness = best_fitness;
                            ts.fitness_history.push(best_fitness);
                            app.best_genome = Some(genome_weights);
                        }
                        TrainingMessage::Finished => {
                            ts.running = false;
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
        AppState::LoadCppChampion => match cpp_compat::load_cpp_champion("fittest.log") {
            Ok(weights) => {
                app.best_genome = Some(weights.clone());
                app.simulation = Some(SimulationState::new(
                    weights.clone(),
                    weights,
                    "C++ Champion".to_string(),
                    "C++ Champion".to_string(),
                ));
                app.state = AppState::Simulation;
                app.error_message = None;
            }
            Err(e) => {
                app.error_message = Some(format!("Failed to load C++ champion: {}", e));
                app.state = AppState::MainMenu;
            }
        },
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
        AppState::LoadCppChampion => {
            // This state is transient, just show a loading message
            let loading_text = "Loading C++ Champion...";
            let paragraph = Paragraph::new(loading_text).alignment(Alignment::Center);
            f.render_widget(paragraph, f.area());
        }
        AppState::Exiting => {} // Do nothing, the app will exit
    };

    if let Some(msg) = &app.error_message {
        components::error_popup::draw_error_popup(f, msg, f.area());
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
