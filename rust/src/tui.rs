use crate::config::{Activation, Engine, EvolutionConfig};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    style::{Color, Modifier, Style},
    symbols,
    widgets::{
        BarChart, Block, Borders, Gauge, List, ListItem, ListState, Paragraph, Sparkline, Tabs,
    },
};
use std::io::{self, stdout, Result};
use std::time::Duration;

/// Application state.
enum AppState {
    MainMenu,
    Training,

    Exiting,
}

/// Main menu state.
struct MainMenu {
    state: ListState,
}

impl Default for MainMenu {
    fn default() -> Self {
        let mut state = ListState::default();
        state.select(Some(0));
        Self { state }
    }
}

/// Central application struct.
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

pub struct TrainingState {
    pub running: bool,
    pub current_generation: usize,
    pub total_generations: usize,
    pub games_completed: usize,

    pub best_fitness: f32,
    pub fitness_history: Vec<f32>,
    pub engine: Engine,
    pub genome_weights: Vec<f32>,
    pub log: Vec<String>,
}

pub struct App {
    state: AppState,
    config: EvolutionConfig,
    main_menu: MainMenu,
    pub training: Option<TrainingState>,
    pub tab: usize,
    pub tx: Option<Sender<TrainingMessage>>,
    pub rx: Option<Receiver<TrainingMessage>>,
}

pub enum TrainingMessage {
    Progress {
        generation: usize,
        games_completed: usize,
        best_fitness: f32,
        genome_weights: Vec<f32>,
    },
    Finished,
    Log(String),
}

impl App {
    fn new() -> Self {
        Self {
            state: AppState::MainMenu,
            config: EvolutionConfig::default(),
            main_menu: MainMenu::default(),
            training: None,
            tab: 0,
            tx: None,
            rx: None,
        }
    }

    /// Runs the main application loop.
    pub fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> Result<()> {
        while !matches!(self.state, AppState::Exiting) {
            terminal.draw(|f| ui(f, self))?;
            self.handle_events()?;

            if matches!(self.state, AppState::Training) && self.training.is_none() {
                // Start training in a background thread with channel communication
                let (tx, rx) = mpsc::channel();
                let config = self.config.clone();
                self.tx = Some(tx.clone());
                self.rx = Some(rx);
                self.training = Some(TrainingState {
                    running: true,
                    current_generation: 0,
                    total_generations: config.generations as usize,
                    games_completed: 0,
                    best_fitness: 0.0,
                    fitness_history: Vec::new(),
                    engine: config.engine,
                    genome_weights: vec![0.0; 217],
                    log: Vec::new(),
                });
                thread::spawn(move || run_training_threaded(config, tx));
            }

            // Poll for training messages and update state
            if let Some(rx) = &self.rx {
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        TrainingMessage::Progress {
                            generation,
                            games_completed,
                            best_fitness,
                            genome_weights,
                        } => {
                            if let Some(training) = &mut self.training {
                                training.current_generation = generation;
                                training.games_completed = games_completed;
                                training.best_fitness = best_fitness;
                                training.fitness_history.push(best_fitness);
                                training.genome_weights = genome_weights;
                            }
                        }
                        TrainingMessage::Log(line) => {
                            if let Some(training) = &mut self.training {
                                training.log.push(line);
                            }
                        }
                        TrainingMessage::Finished => {
                            if let Some(training) = &mut self.training {
                                training.running = false;
                                training.log.push(
                                    "Training finished! Press Enter to return to menu.".into(),
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Handles user input and events.
    fn handle_events(&mut self) -> Result<()> {
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match self.state {
                        AppState::MainMenu => self.handle_main_menu_keys(key.code),
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    /// Handles key presses in the main menu.
    fn handle_main_menu_keys(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') => self.state = AppState::Exiting,
            KeyCode::Down => {
                let selection = self.main_menu.state.selected().unwrap_or(0);
                if selection < 7 {
                    self.main_menu.state.select(Some((selection + 1) as usize));
                }
            }
            KeyCode::Up => {
                let selection = self.main_menu.state.selected().unwrap_or(0);
                if selection > 0 {
                    self.main_menu.state.select(Some((selection - 1) as usize));
                }
            }
            KeyCode::Left => self.modify_config_value(false),
            KeyCode::Right => self.modify_config_value(true),
            KeyCode::Enter => {
                if let Some(sel) = self.main_menu.state.selected() {
                    if sel == 7 {
                        self.state = AppState::Training;
                    }
                }
            }
            _ => {}
        }
    }

    /// Modifies the selected configuration value up or down.
    fn modify_config_value(&mut self, increase: bool) {
        if let Some(selected) = self.main_menu.state.selected() {
            let f_step = 0.01;
            let i_step = 1;
            match selected {
                0 => {
                    // Engine
                    let engines = [
                        Engine::Stack,
                        Engine::Simd,
                        Engine::Heap,
                        Engine::Gpu,
                        Engine::ConcurrentStack,
                        Engine::ConcurrentSimd,
                        Engine::ConcurrentHeap,
                    ];
                    let current_idx = engines
                        .iter()
                        .position(|&e| e as u32 == self.config.engine as u32)
                        .unwrap_or(0);
                    let mut next_idx = if increase {
                        current_idx + 1
                    } else {
                        current_idx.saturating_sub(1)
                    };
                    next_idx %= engines.len();
                    self.config.engine = engines[next_idx];
                    self.config.concurrent = matches!(
                        self.config.engine,
                        Engine::ConcurrentStack | Engine::ConcurrentSimd | Engine::ConcurrentHeap
                    );
                }
                1 => {
                    // Generations
                    if increase {
                        self.config.generations += i_step;
                    } else {
                        self.config.generations = self.config.generations.saturating_sub(i_step);
                    }
                }
                2 => {
                    // Population
                    if increase {
                        self.config.population_size =
                            self.config.population_size.saturating_add(i_step as usize);
                    } else {
                        self.config.population_size =
                            self.config.population_size.saturating_sub(i_step as usize);
                    }
                }
                3 => {
                    // Elites
                    if increase {
                        self.config.elite_count =
                            self.config.elite_count.saturating_add(i_step as usize);
                    } else {
                        self.config.elite_count =
                            self.config.elite_count.saturating_sub(i_step as usize);
                    }
                }
                4 => {
                    // Mutation Rate
                    if increase {
                        self.config.mutation_rate += f_step;
                    } else {
                        self.config.mutation_rate -= f_step;
                    }
                    self.config.mutation_rate = self.config.mutation_rate.clamp(0.0, 1.0);
                }
                5 => {
                    // Mutation Strength
                    if increase {
                        self.config.mutation_strength += f_step;
                    } else {
                        self.config.mutation_strength -= f_step;
                    }
                    self.config.mutation_strength = self.config.mutation_strength.clamp(0.0, 1.0);
                }
                6 => {
                    // Activation
                    let activations = [
                        Activation::Tanh,
                        Activation::Relu,
                        Activation::Atan,
                        Activation::Linear,
                    ];
                    let current_idx = activations
                        .iter()
                        .position(|&a| a == self.config.activation)
                        .unwrap_or(0);
                    let mut next_idx = if increase {
                        current_idx + 1
                    } else {
                        current_idx.saturating_sub(1)
                    };
                    next_idx %= activations.len();
                    self.config.activation = activations[next_idx];
                }
                _ => {}
            }
        }
    }
}

/// Main entry point for the TUI.
pub fn run_app() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new();
    let res = app.run(&mut terminal);
    restore_terminal()?;
    res
}

/// Sets up the terminal for TUI rendering.
fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    let mut stdout = stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

/// Restores the terminal to its original state.
fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}

/// Renders the user interface.
fn ui(f: &mut Frame, app: &mut App) {
    match app.state {
        AppState::MainMenu => main_menu_ui(f, app),
        AppState::Training => {
            // Tab navigation (Dashboard, Logs, Genome)
            let tab_titles = ["Dashboard", "Logs", "Genome"];
            let tab_spans: Vec<Span> = tab_titles
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if i == app.tab {
                        Span::styled(
                            *t,
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        )
                    } else {
                        Span::raw(*t)
                    }
                })
                .collect();
            let tabs = Tabs::new(tab_spans)
                .block(Block::default().borders(Borders::ALL).title("Training"))
                .select(app.tab)
                .highlight_style(Style::default().fg(Color::Yellow));
            let area = f.area();
            let tab_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(0)])
                .split(area);
            f.render_widget(tabs, tab_chunks[0]);

            match app.tab {
                0 => {
                    // Dashboard
                    if let Some(training) = &app.training {
                        let v_chunks = Layout::default()
                            .direction(Direction::Vertical)
                            .constraints([
                                Constraint::Length(4),
                                Constraint::Length(3),
                                Constraint::Length(7),
                                Constraint::Min(6),
                            ])
                            .split(tab_chunks[1]);

                        // Engine badge
                        let engine_color = match training.engine {
                            Engine::Stack => Color::Gray,
                            Engine::Simd => Color::Yellow,
                            Engine::Heap => Color::Blue,
                            Engine::Gpu => Color::Magenta,
                            Engine::ConcurrentStack => Color::Cyan,
                            Engine::ConcurrentSimd => Color::LightYellow,
                            Engine::ConcurrentHeap => Color::LightBlue,
                        };
                        let badge = Paragraph::new(format!("Engine: {}", training.engine.to_str()))
                            .style(
                                Style::default()
                                    .fg(engine_color)
                                    .add_modifier(Modifier::BOLD),
                            )
                            .block(Block::default().borders(Borders::ALL).title("Engine"));
                        f.render_widget(badge, v_chunks[0]);

                        // Progress bar
                        let progress =
                            training.current_generation as f64 / training.total_generations as f64;
                        let gauge = Gauge::default()
                            .block(Block::default().title("Generation Progress"))
                            .gauge_style(Style::default().fg(Color::Green).bg(Color::Black))
                            .label(format!(
                                "{}/{}",
                                training.current_generation, training.total_generations
                            ))
                            .ratio(progress);
                        f.render_widget(gauge, v_chunks[1]);

                        // Sparkline for fitness history
                        let fitness_data: Vec<u64> =
                            training.fitness_history.iter().map(|f| *f as u64).collect();
                        let sparkline = Sparkline::default()
                            .block(Block::default().title("Best Fitness"))
                            .style(Style::default().fg(Color::Cyan))
                            .data(&fitness_data)
                            .bar_set(symbols::bar::NINE_LEVELS);
                        f.render_widget(sparkline, v_chunks[2]);

                        // Genome bar chart (first 50 weights)
                        let genome_data: Vec<(&str, u64)> = training
                            .genome_weights
                            .iter()
                            .take(50)
                            .enumerate()
                            .map(|(_i, w)| ("", (w.abs() * 100.0) as u64))
                            .collect();
                        let barchart = BarChart::default()
                            .block(Block::default().title("Best Genome (first 50 weights)"))
                            .data(&genome_data)
                            .bar_width(1)
                            .bar_gap(0)
                            .bar_style(Style::default().fg(Color::Magenta))
                            .value_style(Style::default().fg(Color::White));
                        f.render_widget(barchart, v_chunks[3]);
                    }
                }
                1 => {
                    // Logs
                    if let Some(training) = &app.training {
                        let log_text = training
                            .log
                            .iter()
                            .rev()
                            .take(15)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("\n");
                        let log_paragraph = Paragraph::new(log_text.as_str());
                        f.render_widget(log_paragraph, tab_chunks[1]);
                    }
                }
                2 => {
                    // Genome (full 217 weights as color blocks)
                    if let Some(training) = &app.training {
                        let weights = &training.genome_weights;
                        let grid_w = 31;
                        let grid_h = 7;
                        let mut grid = vec![vec![Span::raw(" "); grid_w]; grid_h];
                        for (i, w) in weights.iter().take(grid_w * grid_h).enumerate() {
                            let y = i / grid_w;
                            let x = i % grid_w;
                            let shade = if *w > 0.5 {
                                Style::default().bg(Color::Green)
                            } else if *w < -0.5 {
                                Style::default().bg(Color::Red)
                            } else {
                                Style::default().bg(Color::DarkGray)
                            };
                            let cell = Span::styled(" ", shade);
                            grid[y as usize][x as usize] = cell;
                        }
                        let mut lines = Vec::new();
                        for row in grid {
                            lines.push(Line::from(row));
                        }
                        let para = Paragraph::new(lines)
                            .block(Block::default().title("Genome Weights Grid"));
                        f.render_widget(para, tab_chunks[1]);
                    }
                }
                _ => {}
            }
        }
        AppState::Exiting => { /* No UI needed */ }
    }
}

/// Runs the training process based on the provided configuration.
fn run_training_threaded(config: EvolutionConfig, tx: std::sync::mpsc::Sender<TrainingMessage>) {
    // TODO: Integrate with real GA logic. For now, simulate progress.
    let total_generations = config.generations as usize;
    let total_games = config.population_size as usize;

    for gen in 0..total_generations {
        let best_fitness = (gen as f32) + rand::random::<f32>();
        let genome_weights = (0..217).map(|_| rand::random::<f32>()).collect();
        let games_completed = total_games;
        tx.send(TrainingMessage::Progress {
            generation: gen + 1,
            games_completed,
            best_fitness,
            genome_weights,
        })
        .ok();
        tx.send(TrainingMessage::Log(format!(
            "Generation {} complete. Best fitness: {:.2}",
            gen + 1,
            best_fitness
        )))
        .ok();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    tx.send(TrainingMessage::Finished).ok();
}

/// Renders the main menu UI.
fn main_menu_ui(f: &mut Frame, app: &mut App) {
    // Add a color-coded badge for engine type
    let engine_color = match app.config.engine {
        Engine::Stack => Color::Green,
        Engine::Simd => Color::Blue,
        Engine::ConcurrentStack | Engine::ConcurrentSimd => Color::Cyan,
        Engine::Gpu => Color::Magenta,
        Engine::Heap | Engine::ConcurrentHeap => Color::Yellow,
    };
    let engine_badge = Span::styled(
        format!(" {} ", app.config.engine.to_str()),
        Style::default()
            .fg(engine_color)
            .add_modifier(Modifier::BOLD),
    );
    let items = [
        format!("     Engine: < {} >", app.config.engine.to_str()),
        format!("Generations: < {} >", app.config.generations),
        format!(" Population: < {} >", app.config.population_size),
        format!("      Elites: < {} >", app.config.elite_count),
        format!("Mutation Rate: < {:.2} >", app.config.mutation_rate),
        format!(" Mut Strength: < {:.2} >", app.config.mutation_strength),
        format!(" Activation: < {} >", app.config.activation.to_str()),
        "[ Start Training ]".to_string(),
    ];

    let list_items: Vec<ListItem> = items.iter().map(|i| ListItem::new(i.as_str())).collect();

    let list = List::new(list_items)
        .block(Block::default().borders(Borders::ALL).title("Main Menu"))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    let instructions = Paragraph::new("Use ↑/↓ to navigate, ←/→ to change values, Enter to start.")
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center);

    // Render engine badge at the top
    let badge_area = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(f.area());
    f.render_widget(Paragraph::new(engine_badge.clone()), badge_area[0]);
    f.render_stateful_widget(list, badge_area[1], &mut app.main_menu.state);
    f.render_widget(instructions, badge_area[2]);
}
