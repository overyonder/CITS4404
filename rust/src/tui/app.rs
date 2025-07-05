//! Application state and main loop for the TUI.
use crate::config::Config;
use crate::tui::{simulation::SimulationState, training::TrainingState};

use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::Result;
use std::sync::mpsc::{Receiver, Sender};

/// Represents the current high-level state or view of the TUI.
/// The main event loop will render and handle input differently based on this state.
#[derive(Debug, PartialEq)]
pub enum AppState {
    /// The initial view, allowing the user to select a mode.
    MainMenu,
    /// A view for configuring training parameters before starting.
    Configuring,
    /// The view for monitoring and managing the evolutionary training process.
    Training,
    /// The view for visualizing a game of Pong with a trained neural network.
    Simulation,
    /// A transient state for loading a C++ champion.
    LoadCppChampion,
    /// A transient state that signals the main loop to terminate.
    Exiting,
}

/// Holds the state for the configuration editor UI.
#[derive(Default)]
pub struct ConfigEditor {
    /// The index of the currently selected configuration item.
    pub selected_index: usize,
}

/// The core application struct that holds all state for the TUI.
///
/// # Fields
/// - `state`: The current active `AppState`.
/// - `config`: The `EvolutionConfig` used for training and simulation.
/// - `main_menu`: State for the main menu widget.
/// - `training`: State for the training view. `None` if not in training mode.
/// - `simulation`: State for the simulation view. `None` if not in simulation mode.
/// - `tab`: The index of the currently selected tab within a view (if applicable).
/// - `tx`, `rx`: Sender and Receiver for message passing between the main UI thread
///   and a background worker thread (e.g., for running the training process without
///   blocking the UI).
pub struct App {
    pub state: AppState,
    pub config: Config,
    pub main_menu: crate::tui::ui::MainMenu,
    pub config_editor: ConfigEditor,
    pub training: Option<TrainingState>,
    pub simulation: Option<SimulationState>,
    /// The best genome discovered so far, to be used in simulation.
    pub best_genome: Option<Vec<f32>>,

    pub tx: Option<Sender<crate::tui::training::TrainingMessage>>,
    pub rx: Option<Receiver<crate::tui::training::TrainingMessage>>,
    pub error_message: Option<String>,
}

impl App {
    /// Creates a new `App` with default initial state.
    pub fn new() -> Self {
        Self {
            state: AppState::MainMenu,
            config: Config::default(),
            main_menu: crate::tui::ui::MainMenu::default(),
            config_editor: ConfigEditor::default(),
            training: None,
            simulation: None,
            best_genome: None,
            tx: None,
            rx: None,
            error_message: None,
        }
    }

    /// Runs the main TUI event loop.
    ///
    /// This function delegates to the `main_event_loop` in the `ui` module,
    /// which handles drawing and event processing.
    pub fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> Result<()> {
        crate::tui::ui::main_event_loop(self, terminal)
    }
}
