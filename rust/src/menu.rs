use crate::{
    evolution::run_evolution_stack,
    game::GameState,
    net::Net,
    paddle::Paddle,
    tui::render_game,
};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem, ListState},
};
use std::io::{self, stdout, Result};
use std::time::Duration;
use std::fs;

const BEST_NET_FILE: &str = "best_net.json";

/// The main application state.
enum AppState {
    MainMenu,
    Exiting,
}

/// Represents the main menu, managing its state and items.
struct Menu {
    state: ListState,
    items: Vec<&'static str>,
}

impl Menu {
    fn new() -> Self {
        Self {
            state: ListState::default(),
            items: vec![
                "Train Network (Stack)",
                "Train Network (SIMD - placeholder)",
                "Train Network (GPU - placeholder)",
                "Simulate with Best Net",
                "Exit",
            ],
        }
    }

    fn next(&mut self) {
        let i = match self.state.selected() {
            Some(i) => (i + 1) % self.items.len(),
            None => 0,
        };
        self.state.select(Some(i));
    }

    fn previous(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i == 0 {
                    self.items.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
    }
}

/// Sets up the terminal for TUI rendering.
fn setup_terminal() -> io::Result<Terminal<impl Backend>> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout()))
}

/// Restores the terminal to its original state.
fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}

/// Runs the main interactive terminal application.
pub fn run_app() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app_state = AppState::MainMenu;
    let mut menu = Menu::new();
    menu.state.select(Some(0));

    while !matches!(app_state, AppState::Exiting) {
        terminal.draw(|f| {
            let size = f.area();
            let title = "PONG AI - MAIN MENU";

            let list_items: Vec<ListItem> = menu
                .items
                .iter()
                .map(|i| ListItem::new(*i))
                .collect();

            let list = List::new(list_items)
                .block(Block::default().borders(Borders::ALL).title(title))
                .highlight_style(
                    Style::default()
                        .bg(Color::LightGreen)
                        .add_modifier(Modifier::BOLD),
                )
                .highlight_symbol(">> ");

            f.render_stateful_widget(list, size, &mut menu.state);
        })?;

        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => app_state = AppState::Exiting,
                    KeyCode::Down => menu.next(),
                    KeyCode::Up => menu.previous(),
                    KeyCode::Enter => {
                        if let Some(selected) = menu.state.selected() {
                            match selected {
                                0 => { // Train
                                    restore_terminal()?;
                                    run_evolution_stack(100).expect("Training failed");
                                    println!("\nTraining complete. Press Enter to return to the menu.");
                                    io::stdin().read_line(&mut String::new())?;
                                    terminal = setup_terminal()?;
                                }
                                1 | 2 => { /* Placeholders */ }
                                3 => { // Simulate
                                    restore_terminal()?;
                                    match fs::read_to_string(BEST_NET_FILE) {
                                        Ok(json) => {
                                            let best_net: Net = serde_json::from_str(&json)?;
                                            let p1 = Paddle::new_with_net(best_net);
                                            let p2 = Paddle::new_with_net(Net::new());
                                            let game_state = GameState::new_with_paddles(p1, p2);
                                            render_game(Some(game_state))?;
                                        }
                                        Err(e) => {
                                            println!("Could not load best_net.json: {}", e);
                                            println!("Please train a network first.");
                                            println!("\nPress Enter to return to the menu.");
                                            io::stdin().read_line(&mut String::new())?;
                                        }
                                    }
                                    terminal = setup_terminal()?;
                                }
                                4 => app_state = AppState::Exiting, // Exit
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    restore_terminal()?;
    Ok(())
}
