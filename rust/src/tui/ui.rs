//! UI rendering and main event loop for the TUI.
use crate::cpp_compat;
use crate::{
    config::{Activation, Engine, FitnessFunc},
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
    style::{Color, Modifier, Style},
    widgets::canvas::{Canvas, Rectangle},
    widgets::{BarChart, Block, Borders, Clear, List, ListItem, Paragraph, Sparkline, Wrap},
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
                    AppState::Configuring => handle_config_input(app, key.code),
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
    let items_len = 4; // "Train", "Simulate", "Load C++", "Quit"
    match key_code {
        KeyCode::Down => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + 1) % items_len,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        KeyCode::Up => {
            let i = match app.main_menu.state.selected() {
                Some(i) => (i + items_len - 1) % items_len,
                None => 0,
            };
            app.main_menu.state.select(Some(i));
        }
        KeyCode::Enter => {
            if let Some(selected) = app.main_menu.state.selected() {
                match selected {
                    0 => {
                        // "Train New Model" -> Go to Config Screen
                        app.state = AppState::Configuring;
                        app.error_message = None; // Clear any previous errors
                    }
                    1 => {
                        // "Simulate Best Model"
                        if let Some(weights) = app.best_genome.clone() {
                            app.simulation = Some(SimulationState::new(weights));
                            app.state = AppState::Simulation;
                            app.error_message = None;
                        } else {
                            app.error_message = Some("No best genome available to simulate. Please train a model first.".to_string());
                        }
                    }
                    2 => {
                        // "Load C++ Champion"
                        app.state = AppState::LoadCppChampion;
                    }
                    3 => app.state = AppState::Exiting,
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Builds the UI for the current application state.
fn ui_builder(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .constraints([Constraint::Min(0)].as_ref())
        .split(f.area());

    match app.state {
        AppState::MainMenu => draw_main_menu(f, app, chunks[0]),
        AppState::Configuring => draw_config_ui(f, app, chunks[0]),
        AppState::Training => draw_training_ui(f, app, chunks[0]),
        AppState::Simulation => draw_simulation_ui(f, app, chunks[0]),
        _ => {} // Exiting, LoadCppChampion have no UI
    }

    // Draw error popup if there is an error message
    if let Some(error_message) = &app.error_message {
        draw_error_popup(f, error_message.clone(), f.area());
    }
}

/// Draws a popup box with an error message.
fn draw_error_popup(f: &mut Frame, error_message: String, area: Rect) {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(35),
            Constraint::Percentage(30), // Popup height
            Constraint::Percentage(35),
        ])
        .split(area);

    let popup_area = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(60), // Popup width
            Constraint::Percentage(20),
        ])
        .split(popup_layout[1])[1];

    let block = Block::default()
        .title(" Error ")
        .title_style(Style::default().fg(Color::White).bg(Color::Red))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let paragraph = Paragraph::new(error_message)
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(Color::Red))
        .block(block);

    f.render_widget(Clear, popup_area); //this clears the background
    f.render_widget(paragraph, popup_area);
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

    if let (Some(msg), Some(err_area)) = (app.error_message.as_ref(), error_area) {
        let p = Paragraph::new(msg.clone())
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
    if let Some(training_state) = &app.training {
        // Create a two-column layout
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        // Left column for progress and sparkline
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),      // For the gauge
                Constraint::Min(10),        // For the sparkline
                Constraint::Percentage(50), // For the genome chart
            ])
            .split(main_chunks[0]);

        // Right column for info and logs
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(10)]) // Info and Log
            .split(main_chunks[1]);

        // --- Progress Gauge ---
        let progress_percent = if training_state.total_generations > 0 {
            (training_state.current_generation as f64 / training_state.total_generations as f64)
                .min(1.0)
        } else {
            0.0
        };
        let progress_label = format!("{:.0}%", progress_percent * 100.0);
        let gauge = ratatui::widgets::Gauge::default()
            .block(Block::default().title("Progress").borders(Borders::ALL))
            .gauge_style(
                Style::default()
                    .fg(Color::Green)
                    .bg(Color::Black)
                    .add_modifier(Modifier::ITALIC),
            )
            .percent((progress_percent * 100.0) as u16)
            .label(progress_label);
        f.render_widget(gauge, left_chunks[0]);

        // --- Fitness Sparkline ---
        let fitness_data: Vec<u64> = training_state
            .fitness_history
            .iter()
            .map(|&f| f as u64)
            .collect();
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .title("Fitness History")
                    .borders(Borders::ALL),
            )
            .data(&fitness_data)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(sparkline, left_chunks[1]);

        // --- Genome Bar Chart ---
        // TODO: This is a memory leak. The string labels are leaked every frame.
        // A better solution would be to use a widget that can take owned Strings,
        // or to store the labels in the app state.
        let genome_data: Vec<(&str, u64)> = training_state
            .genome_weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i.to_string(), (w.abs() * 100.0) as u64))
            .map(|(s, v)| (&*Box::leak(s.into_boxed_str()), v))
            .collect();

        let barchart = BarChart::default()
            .block(Block::default().title("Champion Genome").borders(Borders::ALL))
            .data(&genome_data)
            .bar_width(1)
            .bar_style(Style::default().fg(Color::LightBlue))
            .value_style(Style::default().fg(Color::Black).bg(Color::LightBlue));
        f.render_widget(barchart, left_chunks[2]);

        // --- Info Panel ---
        let info_block = Block::default().title("Info").borders(Borders::ALL);
        let info_area = info_block.inner(right_chunks[0]);
        f.render_widget(info_block, right_chunks[0]);

        let info_chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .split(info_area);

        let engine_badge = Paragraph::new(format!("Engine: {}", training_state.engine))
            .style(Style::default().bg(Color::Blue).fg(Color::White));
        f.render_widget(engine_badge, info_chunks[0]);

        let concurrent_badge = Paragraph::new(if app.config.concurrent {
            "Mode: Concurrent"
        } else {
            "Mode: Sequential"
        })
        .style(
            Style::default()
                .bg(if app.config.concurrent {
                    Color::Magenta
                } else {
                    Color::Gray
                })
                .fg(Color::White),
        );
        f.render_widget(concurrent_badge, info_chunks[1]);

        let elapsed = training_state.start_time.elapsed();
        let stopwatch = Paragraph::new(format!(
            "Time: {:02}:{:02}.{:03}",
            elapsed.as_secs() / 60,
            elapsed.as_secs() % 60,
            elapsed.as_millis() % 1000
        ));
        f.render_widget(stopwatch, info_chunks[2]);

        // --- Log Panel ---
        let log_messages: Vec<ListItem> = training_state
            .log
            .iter()
            .rev() // Show newest logs at the bottom
            .map(|msg| ListItem::new(msg.as_str()))
            .collect();
        let log_list = List::new(log_messages)
            .block(Block::default().title("Log").borders(Borders::ALL));
        f.render_widget(log_list, right_chunks[1]);
    } else {
        let p = Paragraph::new("No training in progress.")
            .block(Block::default().title("Training").borders(Borders::ALL));
        f.render_widget(p, area);
    }
}

fn draw_simulation_ui(f: &mut Frame, app: &mut App, area: Rect) {
    let sim_block = Block::default()
        .title("Simulation Mode (Press 'q' to quit)")
        .borders(Borders::ALL);
    let inner_area = sim_block.inner(area);
    f.render_widget(sim_block.clone(), area);

    if let Some(sim) = &app.simulation {
        let canvas = Canvas::default()
            .block(Block::default())
            .x_bounds([0.0, WIDTH as f64])
            .y_bounds([0.0, HEIGHT as f64])
            .paint(|ctx| {
                // Draw the ball
                ctx.draw(&Rectangle {
                    x: sim.game.ball_pos.0 as f64,
                    y: sim.game.ball_pos.1 as f64,
                    width: 1.0,
                    height: 1.0,
                    color: Color::Yellow,
                });

                // Draw left paddle
                ctx.draw(&Rectangle {
                    x: 0.0,
                    y: sim.game.paddle1_pos as f64,
                    width: 1.0,
                    height: PADDLE_HEIGHT as f64,
                    color: Color::Blue,
                });

                // Draw right paddle
                ctx.draw(&Rectangle {
                    x: (WIDTH - 1) as f64,
                    y: sim.game.paddle2_pos as f64,
                    width: 1.0,
                    height: PADDLE_HEIGHT as f64,
                    color: Color::Red,
                });
            });
        f.render_widget(canvas, inner_area);
    } else {
        let paragraph = Paragraph::new("No trained genome available. Please run training first.")
            .block(sim_block)
            .wrap(Wrap { trim: true });
        f.render_widget(paragraph, area);
    }
}

/// Draws the UI for the configuration editor.
fn draw_config_ui(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(90), Constraint::Percentage(10)])
        .split(area);

    let config_items = vec![
        format!("Engine             : {}", app.config.engine),
        format!("Generations        : {}", app.config.generations),
        format!("Population Size    : {}", app.config.population_size),
        format!("Elite Count        : {}", app.config.elite_count),
        format!("Mutation Rate      : {:.2}", app.config.mutation_rate),
        format!("Mutation Strength  : {:.2}", app.config.mutation_strength),
        format!("Activation         : {}", app.config.activation),
        format!("Concurrent         : {}", app.config.concurrent),
        format!("Fitness Func       : {}", app.config.fitness_func),
    ];

    let items: Vec<ListItem> = config_items
        .iter()
        .map(|i| ListItem::new(i.as_str()))
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Training Configuration"),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    let mut list_state = ratatui::widgets::ListState::default();
    list_state.select(Some(app.config_editor.selected_index));

    f.render_stateful_widget(list, chunks[0], &mut list_state);

    let help_text = "Use ↑/↓ to navigate, ←/→ to change values. Press Enter to start training, Q to quit.";
    let help_paragraph = Paragraph::new(help_text)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Help"));
    f.render_widget(help_paragraph, chunks[1]);
}

/// Handles user input on the configuration screen.
fn handle_config_input(app: &mut App, key_code: KeyCode) {
    let num_items = 9; // Number of configurable items
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.state = AppState::MainMenu;
        }
        KeyCode::Up => {
            let i = &mut app.config_editor.selected_index;
            *i = i.saturating_sub(1);
        }
        KeyCode::Down => {
            let i = &mut app.config_editor.selected_index;
            if *i < num_items - 1 {
                *i += 1;
            }
        }
        KeyCode::Left => change_config_value(app, false),
        KeyCode::Right => change_config_value(app, true),
        KeyCode::Enter => {
            // Start training with the configured settings.
            // `evolve_with_progress` will spawn a background thread.
            app.state = AppState::Training;
            let (tx, rx) = mpsc::channel();
            app.tx = Some(tx.clone());
            app.rx = Some(rx);
            app.training = Some(TrainingState::new(&app.config));
            let config = app.config;
            evolve_with_progress(config, tx);
        }
        _ => {}
    }
}

/// Helper function to modify a configuration value based on the selected index.
fn change_config_value(app: &mut App, increase: bool) {
    let config = &mut app.config;
    match app.config_editor.selected_index {
        0 => { // Engine
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
        1 => { // Generations
            let step = 10;
            if increase {
                config.generations += step;
            } else {
                config.generations = config.generations.saturating_sub(step);
            }
        }
        2 => { // Population Size
            let step = 10;
            if increase {
                config.population_size += step;
            } else {
                config.population_size = config.population_size.saturating_sub(step);
            }
        }
        3 => { // Elite Count
            if increase {
                config.elite_count += 1;
            } else {
                config.elite_count = config.elite_count.saturating_sub(1);
            }
        }
        4 => { // Mutation Rate
            let step = 0.01;
            if increase {
                config.mutation_rate += step;
            } else {
                config.mutation_rate = (config.mutation_rate - step).max(0.0);
            }
        }
        5 => { // Mutation Strength
            let step = 0.01;
            if increase {
                config.mutation_strength += step;
            } else {
                config.mutation_strength = (config.mutation_strength - step).max(0.0);
            }
        }
        6 => { // Activation
            let current_activation_index = match config.activation {
                Activation::Tanh => 0,
                Activation::Relu => 1,
                Activation::Atan => 2,
                Activation::Linear => 3,
            };
            let next_index = if increase {
                (current_activation_index + 1) % 4
            } else {
                (current_activation_index + 3) % 4
            };
            config.activation = match next_index {
                0 => Activation::Tanh,
                1 => Activation::Relu,
                2 => Activation::Atan,
                _ => Activation::Linear,
            };
        }
        7 => { // Concurrent
            config.concurrent = !config.concurrent;
        }
        8 => { // Fitness Function
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
        _ => {}
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
