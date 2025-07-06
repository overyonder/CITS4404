//! The main menu component.

use crate::tui::{
    app::{App, AppState, SimulationSetupState},
    model_loader,
    simulation::SimulationState,
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType, Borders, List, ListItem},
    Frame,
};
use std::path::Path;

/// State for the main menu.
pub struct MainMenu {
    pub state: ratatui::widgets::ListState,
}

impl Default for MainMenu {
    fn default() -> Self {
        let mut state = ratatui::widgets::ListState::default();
        state.select(Some(0));
        Self { state }
    }
}

/// Handles user input on the main menu.
pub fn handle_main_menu_input(app: &mut App, key_code: KeyCode) {
    let num_items = 5;
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.state = AppState::Exiting;
        }
        KeyCode::Down => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + 1) % num_items,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        KeyCode::Up => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + num_items - 1) % num_items,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        KeyCode::Enter => {
            if let Some(selected) = app.main_menu.state.selected() {
                match selected {
                    0 => {
                        // "Train New Model"
                        app.state = AppState::Configuring;
                        app.error_message = None;
                    }
                    1 => {
                        // "Simulate Last Champion"
                        if let Some(genome) = app.best_genome.clone() {
                            app.simulation = Some(SimulationState::new(
                                genome.clone(),
                                genome,
                                "Trained Champion".to_string(),
                                "Trained Champion".to_string(),
                            ));
                            app.state = AppState::Simulation;
                        } else {
                            app.error_message =
                                Some("No champion available to simulate.".to_string());
                        }
                    }
                    2 => {
                        // "Simulate from File"
                        match model_loader::load_models_from_dir(Path::new("models")) {
                            Ok(models) => {
                                if models.is_empty() {
                                    app.error_message = Some(
                                        "No models found in the 'models' directory.".to_string(),
                                    );
                                } else {
                                    app.simulation_setup = Some(SimulationSetupState::new(models));
                                    app.state = AppState::SimulationSetup;
                                    app.error_message = None;
                                }
                            }
                            Err(e) => {
                                app.error_message = Some(format!("Failed to load models: {}", e));
                            }
                        }
                    }
                    3 => {
                        // "Load C++ Champion"
                        app.state = AppState::LoadCppChampion;
                        app.error_message = None;
                    }
                    4 => {
                        // "Exit"
                        app.state = AppState::Exiting;
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Draws the UI for the main menu.
pub fn draw_main_menu(f: &mut Frame, app: &mut App, area: Rect) {
    let items = [
        ListItem::new("Train New Model"),
        ListItem::new("Simulate Last Champion"),
        ListItem::new("Simulate from File"),
        ListItem::new("Load C++ Champion"),
        ListItem::new("Exit"),
    ];
    let list = List::new(items)
        .block(
            Block::default()
                .title("Main Menu")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");
    f.render_stateful_widget(list, area, &mut app.main_menu.state);
}
