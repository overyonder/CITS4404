//! The main menu component.

use crate::{
    cpp_compat,
    tui::{
        app::{App, AppState, ModelInfo, SimulationSetupState},
        model_loader,
    },
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState},
    Frame,
};
use std::{fs, io, path::Path};

/// State for the main menu.
pub struct MainMenu {
    pub state: ListState,
}

impl Default for MainMenu {
    fn default() -> Self {
        let mut state = ListState::default();
        state.select(Some(0));
        Self { state }
    }
}

/// Handles user input on the main menu.
pub fn handle_main_menu_input(app: &mut App, key_code: KeyCode) {
    let num_items = 3;
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
                        // "Simulate"
                        match load_models_from_dir(Path::new("models")) {
                            Ok(models) => {
                                if models.is_empty() {
                                    app.error_message = Some(
                                        "No models found in 'models/'. Train a model first.".to_string(),
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
                    2 => {
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

/// Scans the 'models' directory and loads the metadata for each valid model file.
fn load_models_from_dir(dir: &Path) -> io::Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let is_cpp = path.extension().map_or(false, |ext| ext == "log");
        if is_cpp {
            if let Ok((_weights, config)) = cpp_compat::load_cpp_champion(&path.to_string_lossy()) {
                models.push(ModelInfo {
                    path,
                    config,
                    is_cpp: true,
                });
            }
        } else if let Ok((_weights, mut config)) = model_loader::load_model_from_file(&path) {
            // If the model name isn't in the config, use the filename.
            if config.name.is_none() {
                config.name = path.file_stem().map(|s| s.to_string_lossy().to_string());
            }
            models.push(ModelInfo {
                path,
                config,
                is_cpp: false,
            });
        }
    }
    Ok(models)
}

/// Draws the UI for the main menu.
pub fn draw_main_menu(f: &mut Frame, app: &mut App, area: Rect) {
    let items = [
        ListItem::new("Train New Model"),
        ListItem::new("Simulate"),
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
