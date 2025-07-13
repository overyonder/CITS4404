//! The main menu component with enhanced user experience and educational information.
//!
//! # Teaching Note: TUI Design Principles
//! This component demonstrates effective TUI design:
//! - Clear navigation instructions
//! - Informative descriptions for each option
//! - Graceful error handling with user-friendly messages
//! - Consistent visual styling and color coding

use crate::{
    tui::{
        app::{App, AppState, ModelInfo, SimulationSetupState},
        model_loader,
    },
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};
use std::{fs, io, path::Path};

/// State for the main menu with enhanced functionality.
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

/// Enhanced menu items with descriptions and educational context.
const MENU_ITEMS: &[(
    &str,                    // Title
    &str,                    // Description
    &str,                    // Educational note
)] = &[
    (
        "🔬 Train New Model",
        "Create and evolve a new neural network using genetic algorithms",
        "Genetic algorithms mimic natural evolution to optimize neural networks"
    ),
    (
        "🎮 Play",
        "Watch trained models play Pong, compare strategies, or play yourself",
        "Test your evolved AI agents in real-time gameplay scenarios and human vs AI matches"
    ),
    (
        "❌ Exit",
        "Close the application and return to the terminal",
        "All trained models will be preserved in the models/ directory"
    ),
];

/// Handles user input on the main menu with improved navigation.
///
/// # Teaching Note: Input Handling Best Practices
/// - Provides multiple ways to navigate (arrow keys, vim-style keys)
/// - Clear exit options (q, Esc)
/// - Immediate feedback for invalid selections
/// - Graceful error handling with informative messages
pub fn handle_main_menu_input(app: &mut App, key_code: KeyCode) {
    let num_items = MENU_ITEMS.len();
    
    match key_code {
        // Exit options
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
            app.state = AppState::Exiting;
        }
        
        // Navigation - Arrow keys
        KeyCode::Down | KeyCode::Char('j') => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + 1) % num_items,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        KeyCode::Up | KeyCode::Char('k') => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + num_items - 1) % num_items,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        
        // Selection
        KeyCode::Enter | KeyCode::Char(' ') => {
            if let Some(selected) = app.main_menu.state.selected() {
                handle_menu_selection(app, selected);
            }
        }
        
        // Direct shortcuts
        KeyCode::Char('1') => {
            app.main_menu.state.select(Some(0));
            handle_menu_selection(app, 0);
        }
        KeyCode::Char('2') => {
            app.main_menu.state.select(Some(1));
            handle_menu_selection(app, 1);
        }
        KeyCode::Char('3') => {
            app.main_menu.state.select(Some(2));
            handle_menu_selection(app, 2);
        }
        
        _ => {}
    }
}

/// Handles the actual menu selection with enhanced error reporting.
fn handle_menu_selection(app: &mut App, selected: usize) {
    match selected {
        0 => {
            // "Train New Model"
            app.state = AppState::Configuring;
            app.error_message = None;
        }
        1 => {
            // "Play" - Enhanced model loading with better error messages
            match load_models_from_dir(Path::new("models")) {
                Ok(models) => {
                    if models.is_empty() {
                        app.error_message = Some(
                            "No trained models found in 'models/' directory.\n\n\
                            Train a model first using 'Train New Model', or check if you have \
                            .json model files in the models/ folder.".to_string(),
                        );
                    } else {
                        app.simulation_setup = Some(SimulationSetupState::new(models.clone()));
                        app.state = AppState::SimulationSetup;
                        app.error_message = None;
                    }
                }
                Err(e) => {
                    app.error_message = Some(format!(
                        "Failed to load models from 'models/' directory:\n\n{}\n\n\
                        Make sure the models/ directory exists and contains valid .json model files.",
                        e
                    ));
                }
            }
        }
        2 => {
            // "Exit"
            app.state = AppState::Exiting;
        }
        _ => {
            app.error_message = Some("Invalid menu selection. Please try again.".to_string());
        }
    }
}

/// Enhanced model loading with better validation and error reporting.
///
/// # Teaching Note: File I/O Best Practices
/// - Comprehensive error handling for different failure modes
/// - Clear distinction between different file types (JSON vs C++ log files)
/// - Graceful handling of corrupted or invalid files
/// - Informative error messages for debugging
fn load_models_from_dir(dir: &Path) -> io::Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    let mut errors = Vec::new();
    
    // Ensure directory exists
    if !dir.exists() {
        fs::create_dir_all(dir)?;
        return Ok(models); // Return empty list for new directory
    }
    
    // Process each file in the directory
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if !path.is_file() {
            continue;
        }

        // Skip legacy C++ format files (.log files) since compatibility layer is removed
        let is_cpp = path.extension().map_or(false, |ext| ext == "log");
        if is_cpp {
            continue; // Skip C++ files since compatibility layer is removed
        }
        // Handle modern JSON format 
        else if path.extension().map_or(false, |ext| ext == "json") {
            match model_loader::load_model_from_file(&path) {
                Ok((_weights, mut config)) => {
                    // Set model name from filename if not present
                    if config.name.is_none() {
                        config.name = path.file_stem().map(|s| s.to_string_lossy().to_string());
                    }
                    models.push(ModelInfo {
                        path,
                        config,
                    });
                }
                Err(e) => {
                    errors.push(format!("Failed to load JSON model {:?}: {}", path.file_name(), e));
                }
            }
        }
        // Skip other file types silently
    }
    
    // Log errors but don't fail the entire operation
    if !errors.is_empty() {
        tracing::warn!("Some model files could not be loaded: {}", errors.join(", "));
    }
    
    Ok(models)
}

/// Draws an enhanced main menu with better visual design and information.
///
/// # Teaching Note: TUI Layout Design
/// - Uses proper spacing and alignment for visual hierarchy
/// - Color coding for different types of information
/// - Clear separation between interactive elements and help text
/// - Responsive layout that adapts to terminal size
pub fn draw_main_menu(f: &mut Frame, app: &mut App, area: Rect) {
    // Create layout with space for menu, descriptions, and help
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(MENU_ITEMS.len() as u16 + 4), // Menu items + borders
            Constraint::Min(8),                               // Description area
            Constraint::Length(6),                            // Help/controls
        ])
        .split(area);

    // Render main menu list
    draw_menu_list(f, app, chunks[0]);
    
    // Render description panel
    draw_description_panel(f, app, chunks[1]);
    
    // Render help panel
    draw_help_panel(f, chunks[2]);
}

/// Draws the main menu list with enhanced styling.
fn draw_menu_list(f: &mut Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = MENU_ITEMS
        .iter()
        .enumerate()
        .map(|(i, (title, _desc, _note))| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{}. ", i + 1),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    *title,
                    Style::default().fg(Color::White),
                ),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .title("🧠 Pong Neural Network Evolution Trainer")
                .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::White)),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Blue)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    f.render_stateful_widget(list, area, &mut app.main_menu.state);
}

/// Draws an informative description panel for the selected menu item.
fn draw_description_panel(f: &mut Frame, app: &App, area: Rect) {
    let selected_idx = app.main_menu.state.selected().unwrap_or(0);
    let (title, description, educational_note) = MENU_ITEMS[selected_idx];
    
    let text = vec![
        Line::from(vec![
            Span::styled("Description: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(description, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("💡 Teaching Note: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(educational_note, Style::default().fg(Color::Gray)),
        ]),
    ];

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title(format!("📖 About: {}", title.chars().skip_while(|c| !c.is_alphabetic()).collect::<String>()))
                .title_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Green)),
        )
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
}

/// Draws a help panel with navigation instructions.
fn draw_help_panel(f: &mut Frame, area: Rect) {
    let help_text = vec![
        Line::from(vec![
            Span::styled("Navigation: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("↑/↓ or j/k to navigate  "),
            Span::styled("Enter/Space", Style::default().fg(Color::Yellow)),
            Span::raw(" to select  "),
            Span::styled("q/Esc", Style::default().fg(Color::Red)),
            Span::raw(" to exit"),
        ]),
        Line::from(vec![
            Span::styled("Shortcuts: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Press "),
            Span::styled("1", Style::default().fg(Color::Yellow)),
            Span::raw(", "),
            Span::styled("2", Style::default().fg(Color::Yellow)),
            Span::raw(", or "),
            Span::styled("3", Style::default().fg(Color::Yellow)),
            Span::raw(" for direct selection"),
        ]),
    ];

    let paragraph = Paragraph::new(help_text)
        .block(
            Block::default()
                .title("⌨️  Controls")
                .title_style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Magenta)),
        );

    f.render_widget(paragraph, area);
}
