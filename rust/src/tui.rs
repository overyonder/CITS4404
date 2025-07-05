use crate::{constants::*, gamestate::GameState};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};
use std::{

    io::{self, stdout, Result},
    time::Duration,
};

/// The main application state.
enum AppState {
    MainMenu,
    Game,
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
pub fn run_app(generations: u32) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app_state = AppState::MainMenu;
    let mut menu = Menu::new();
    menu.state.select(Some(0));
    let mut best_net_for_sim: Option<crate::individual::Individual> = None;

    while !matches!(app_state, AppState::Exiting) {
        match app_state {
            AppState::MainMenu => {
                terminal.draw(|f| {
                    let size = f.area();
                    let title = "PONG AI - MAIN MENU";
                    let list_items: Vec<ListItem> =
                        menu.items.iter().map(|i| ListItem::new(*i)).collect();
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
                        if key.kind == KeyEventKind::Release {
                            continue;
                        }
                        match key.code {
                            KeyCode::Char('q') => app_state = AppState::Exiting,
                            KeyCode::Down => menu.next(),
                            KeyCode::Up => menu.previous(),
                            KeyCode::Enter => {
                                if let Some(selected) = menu.state.selected() {
                                    match selected {
                                        0 => { // Train
                                            restore_terminal()?;
                                            let mut population = crate::population::Population::abiogenesis();
                                            population.evolve(generations);
                                            println!("\nTraining complete. Press Enter to return to the menu.");
                                            io::stdin().read_line(&mut String::new())?;
                                            terminal = setup_terminal()?;
                                        }
                                        1 | 2 => {} // Placeholders
                                        3 => { // Simulate
                                            restore_terminal()?;
                                            match crate::individual::Individual::load(BEST_NET_FILE) {
                                                Ok(best_net) => {
                                                    best_net_for_sim = Some(best_net);
                                                    app_state = AppState::Game;
                                                }
                                                Err(e) => {
                                                    println!("Could not load {}: {}. Please train a network first.", BEST_NET_FILE, e);
                                                    println!("\nPress Enter to return to the menu.");
                                                    io::stdin().read_line(&mut String::new())?;
                                                    terminal = setup_terminal()?;
                                                }
                                            }
                                        }
                                        4 => app_state = AppState::Exiting,
                                        _ => {}
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            AppState::Game => {
                if let Some(net) = best_net_for_sim {
                    run_game(&mut terminal, &net, &mut app_state)?;
                    // After game returns, restore terminal for menu
                    terminal = setup_terminal()?;
                } else {
                    // This case should not be reachable if logic is correct
                    app_state = AppState::MainMenu;
                }
            }
            AppState::Exiting => break,
        }
    }

    restore_terminal()?;
    Ok(())
}

/// Runs the interactive game simulation loop.
fn run_game(terminal: &mut Terminal<impl Backend>, net: &crate::individual::Individual, app_state: &mut AppState) -> Result<()> {
    let mut game_state = GameState::new();

    loop {
        terminal.draw(|f| ui(f, &game_state))?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Char('q') = key.code {
                    *app_state = AppState::MainMenu;
                    break;
                }
            }
        }

        game_state.tick(net, net);
    }

    Ok(())
}

/// Defines the UI layout and widgets for rendering the game state.
fn ui(frame: &mut Frame, game: &GameState) {
    // Create a layout to split the screen into a score area and a game area.
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // 1-line-high area for scores
            Constraint::Min(0),    // The rest of the screen for the game
        ])
        .split(frame.area());

    let score_area = main_layout[0];
    let game_area = main_layout[1];

    // Render the score paragraph.
    // NOTE: This assumes `game.score` is `[u32; 2]`. Adjust if this is incorrect.
    let score_text = format!("Score: {} - {}", game.scores.0, game.scores.1);
    let score_paragraph = Paragraph::new(score_text).alignment(Alignment::Center);
    frame.render_widget(score_paragraph, score_area);

    // Render the game area block.
    let game_block = Block::default().borders(Borders::ALL).title("Pong");
    let game_area_inner = game_block.inner(game_area); // Get the inner rect to draw in.
    frame.render_widget(game_block, game_area);

    // Draw Ball
    // Convert ball's game coordinates to terminal cell coordinates.
    let ball_x_fraction = game.ball_pos.0 / WIDTH as f32;
    let ball_y_fraction = game.ball_pos.1 / LENGTH as f32;

    let ball_x = game_area_inner.x + (ball_x_fraction * game_area_inner.width as f32) as u16;
    let ball_y = game_area_inner.y + (ball_y_fraction * game_area_inner.height as f32) as u16;

    // Create a 1x1 Rect for the ball.
    let ball_rect = Rect::new(ball_x, ball_y, 1, 1);

    // Render the ball character.
    frame.render_widget(Paragraph::new("●"), ball_rect);

    // Draw Paddles
    // Left Paddle
    let left_paddle_y_fraction = game.paddle1_pos / LENGTH as f32;
    let left_paddle_y = game_area_inner.y + (left_paddle_y_fraction * game_area_inner.height as f32) as u16;
    let left_paddle_rect = Rect::new(
        game_area_inner.x,
        left_paddle_y.saturating_sub(PADDLE_HEIGHT / 2),
        PADDLE_WIDTH as u16,
        PADDLE_HEIGHT as u16,
    );
    frame.render_widget(Paragraph::new("█"), left_paddle_rect);

    // Right Paddle
    let right_paddle_y_fraction = game.paddle2_pos / LENGTH as f32;
    let right_paddle_y = game_area_inner.y + (right_paddle_y_fraction * game_area_inner.height as f32) as u16;
    let right_paddle_rect = Rect::new(
        game_area_inner.right() - PADDLE_WIDTH as u16,
        right_paddle_y.saturating_sub(PADDLE_HEIGHT / 2),
        PADDLE_WIDTH as u16,
        PADDLE_HEIGHT as u16,
    );
    frame.render_widget(Paragraph::new("█"), right_paddle_rect);
}
