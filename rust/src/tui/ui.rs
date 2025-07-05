//! UI rendering and main event loop for the TUI.
use crate::cpp_compat;
use crate::{
    constants::{HEIGHT, PADDLE_HEIGHT, WIDTH},
    tui::{
        app::{App, AppState},
        simulation::SimulationState,
        training::{evolve_with_progress, TrainingMessage, TrainingState},
    },
};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols,
    widgets::canvas::{Canvas, Rectangle},
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline, Wrap},
    Frame, Terminal,
};
use std::{
    io::{Result, Stdout},
    sync::mpsc,
    time::Duration,
};

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
                if let Some(training_state) = app.training.as_mut() {
                    match msg {
                        TrainingMessage::Progress {
                            generation,
                            best_fitness,
                            genome_weights,
                            ..
                        } => {
                            training_state.current_generation = generation;
                            training_state.best_fitness = best_fitness;
                            training_state.fitness_history.push(best_fitness);
                            app.best_genome = Some(genome_weights);
                        }
                        TrainingMessage::Finished => {
                            training_state.running = false;
                        }
                        TrainingMessage::Log(log_msg) => {
                            training_state.log.push(log_msg);
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
                app.simulation = Some(SimulationState::new(weights));
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
    if event::poll(Duration::from_millis(33))? {
        // ~30 FPS
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                match app.state {
                    AppState::MainMenu => handle_main_menu_input(app, key.code),
                    AppState::Training => {
                        if key.code == KeyCode::Char('q') {
                            app.state = AppState::MainMenu;
                            app.training = None;
                            app.tx = None;
                            app.rx = None;
                        }
                    }
                    AppState::Simulation => {
                        if key.code == KeyCode::Char('q') {
                            app.state = AppState::MainMenu;
                            app.simulation = None;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn handle_main_menu_input(app: &mut App, key_code: KeyCode) {
    let menu_items_count = 4;
    match key_code {
        KeyCode::Up => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + menu_items_count - 1) % menu_items_count,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        KeyCode::Down => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + 1) % menu_items_count,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        KeyCode::Enter => match app.main_menu.state.selected() {
            Some(0) => {
                app.state = AppState::Training;
                let (tx, rx) = mpsc::channel();
                app.tx = Some(tx.clone());
                app.rx = Some(rx);
                app.training = Some(TrainingState::new(&app.config));
                evolve_with_progress(app.config.clone(), tx);
            }
            Some(1) => {
                if let Some(genome) = &app.best_genome {
                    app.simulation = Some(SimulationState::new(genome.clone()));
                    app.state = AppState::Simulation;
                    app.error_message = None;
                } else {
                    app.error_message =
                        Some("No model available. Train or load a model first.".to_string());
                }
            }
            Some(2) => {
                app.state = AppState::LoadCppChampion;
            }
            Some(3) => app.state = AppState::Exiting,
            _ => {}
        },
        _ => {}
    }
}

/// Builds the UI for the current application state.
fn ui_builder(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Percentage(100)].as_ref())
        .split(f.area());

    match app.state {
        AppState::MainMenu => draw_main_menu(f, app, chunks[0]),
        AppState::Training => draw_training_ui(f, app, chunks[0]),
        AppState::Simulation => draw_simulation_ui(f, app, chunks[0]),
        AppState::LoadCppChampion => {
            let p = Paragraph::new("Loading C++ champion...")
                .block(Block::default().title("Loading").borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
        AppState::Exiting => {} // Do nothing, will exit loop
    }
}

fn draw_main_menu(f: &mut Frame, app: &mut App, area: Rect) {
    let (menu_area, error_area) = if app.error_message.is_some() {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)])
            .split(area);
        (chunks[0], Some(chunks[1]))
    } else {
        (area, None)
    };

    if let (Some(msg), Some(err_area)) = (app.error_message.take(), error_area) {
        let p = Paragraph::new(msg)
            .style(Style::default().fg(Color::Red))
            .block(Block::default().title("Error").borders(Borders::ALL));
        f.render_widget(p, err_area);
    }

    let items = [
        ListItem::new("Train New Model"),
        ListItem::new("Simulate Best Model"),
        ListItem::new("Load C++ Champion"),
        ListItem::new("Quit"),
    ];
    let list = List::new(items)
        .block(Block::default().title("Main Menu").borders(Borders::ALL))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");
    f.render_stateful_widget(list, menu_area, &mut app.main_menu.state);
}

fn draw_training_ui(f: &mut Frame, app: &mut App, area: Rect) {
    if let Some(state) = &app.training {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(5),
            ])
            .split(area);

        let status_text = format!(
            "Engine: {} | Gen: {}/{} | Best Fitness: {:.2}",
            state.engine.to_string(),
            state.current_generation,
            state.total_generations,
            state.best_fitness
        );
        let status_paragraph = Paragraph::new(status_text).block(
            Block::default()
                .title("Training Status")
                .borders(Borders::ALL),
        );
        f.render_widget(status_paragraph, chunks[0]);

        let log_items: Vec<ListItem> = state
            .log
            .iter()
            .map(|msg| ListItem::new(msg.as_str()))
            .collect();
        let log_list =
            List::new(log_items).block(Block::default().title("Log").borders(Borders::ALL));
        f.render_widget(log_list, chunks[1]);

        let sparkline_data: Vec<u64> = state.fitness_history.iter().map(|&f| f as u64).collect();
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .title("Fitness History")
                    .borders(Borders::ALL),
            )
            .data(&sparkline_data)
            .style(Style::default().fg(Color::Green));
        f.render_widget(sparkline, chunks[2]);
    } else {
        let block = Block::default()
            .title("Training Mode")
            .borders(Borders::ALL);
        let paragraph = Paragraph::new("Initializing training...").block(block);
        f.render_widget(paragraph, area);
    }
}

fn draw_simulation_ui(f: &mut Frame, app: &mut App, area: Rect) {
    let sim_block = Block::default()
        .title("Simulation Mode (Press 'q' to quit)")
        .borders(Borders::ALL);

    if let Some(state) = &app.simulation {
        let game = &state.game;
        let score_text = format!("{} - {}", game.scores.0, game.scores.1);

        let canvas = Canvas::default()
            .block(sim_block)
            .marker(symbols::Marker::Block)
            .x_bounds([0.0, WIDTH as f64])
            .y_bounds([0.0, HEIGHT as f64])
            .paint(move |ctx| {
                // Draw Paddles
                ctx.draw(&Rectangle {
                    x: 0.0,
                    y: game.paddle1_pos as f64,
                    width: 5.0, // Paddle width
                    height: PADDLE_HEIGHT as f64,
                    color: Color::Cyan,
                });
                ctx.draw(&Rectangle {
                    x: (WIDTH - 5) as f64,
                    y: game.paddle2_pos as f64,
                    width: 5.0, // Paddle width
                    height: PADDLE_HEIGHT as f64,
                    color: Color::Red,
                });

                // Draw Ball
                ctx.draw(&Rectangle {
                    x: game.ball_pos.0 as f64,
                    y: game.ball_pos.1 as f64,
                    width: 2.0,
                    height: 2.0,
                    color: Color::White,
                });

                // Draw Score
                ctx.print(
                    WIDTH as f64 / 2.0 - 5.0,
                    HEIGHT as f64 - 5.0,
                    score_text.clone().fg(Color::White),
                );
            });
        f.render_widget(canvas, area);
    } else {
        let paragraph = Paragraph::new("No trained genome available. Please run training first.")
            .block(sim_block)
            .wrap(Wrap { trim: true });
        f.render_widget(paragraph, area);
    }
}

/// Entrypoint for the TUI from main.rs
pub fn run_app() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new();
    let res = app.run(&mut terminal);
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
