//! Application state and main loop for the TUI.
use crate::config::Config;
use crate::tui::{simulation::SimulationState, training::TrainingState};

use ratatui::widgets::TableState;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

/// Represents the current high-level state or view of the TUI.
/// The main event loop will render and handle input differently based on this state.
#[derive(Debug, PartialEq)]
pub enum AppState {
    /// The initial view, allowing the user to select a mode.
    MainMenu,
    /// A view for selecting models for a simulation.
    SimulationSetup,
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

/// Represents the different tabs available in the Training view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tab {
    Generations,
    Matchups,
}

/// Holds metadata for a single, loadable simulation model.
#[derive(Clone)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub config: Config,
}

/// Holds the state for the simulation model selection UI.
pub struct SimulationSetupState {
    pub models: Vec<ModelInfo>,
    pub left_paddle_state: TableState,
    pub right_paddle_state: TableState,
    /// 0 for left table, 1 for right table.
    pub active_table: usize,
}

impl SimulationSetupState {
    /// Moves the selection in the active table to the next item.
    pub fn next(&mut self) {
        let table_state = if self.active_table == 0 {
            &mut self.left_paddle_state
        } else {
            &mut self.right_paddle_state
        };
        let i = match table_state.selected() {
            Some(i) => {
                if i >= self.models.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        table_state.select(Some(i));
    }

    /// Moves the selection in the active table to the previous item.
    pub fn previous(&mut self) {
        let table_state = if self.active_table == 0 {
            &mut self.left_paddle_state
        } else {
            &mut self.right_paddle_state
        };
        let i = match table_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.models.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        table_state.select(Some(i));
    }

    /// Switches the active table between the left and right paddle selectors.
    pub fn switch_focus(&mut self) {
        self.active_table = 1 - self.active_table; // Toggles between 0 and 1
    }

    pub fn new(models: Vec<ModelInfo>) -> Self {
        let mut left_paddle_state = TableState::default();
        if !models.is_empty() {
            left_paddle_state.select(Some(0));
        }
        let mut right_paddle_state = TableState::default();
        if !models.is_empty() {
            right_paddle_state.select(Some(0));
        }

        Self {
            models,
            left_paddle_state,
            right_paddle_state,
            active_table: 0,
        }
    }
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
    pub main_menu: crate::tui::components::main_menu::MainMenu,
    pub config_editor: ConfigEditor,
    pub training: Option<TrainingState>,
    pub simulation: Option<SimulationState>,
    pub simulation_setup: Option<SimulationSetupState>,
    /// The best genome discovered so far, to be used in simulation.
    pub best_genome: Option<Vec<f32>>,
    /// The active tab in the training view.
    pub active_tab: Tab,

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
            main_menu: crate::tui::components::main_menu::MainMenu::default(),
            config_editor: ConfigEditor::default(),
            training: None,
            simulation: None,
            simulation_setup: None,
            best_genome: None,
            active_tab: Tab::Generations,
            tx: None,
            rx: None,
            error_message: None,
        }
    }

    /// Runs the main TUI event loop.
    ///
    /// This function delegates to the `main_event_loop` in the `ui` module,
    /// which handles drawing and event processing.
    /// Switches to the next tab in the training view.
    pub fn next_tab(&mut self) {
        self.active_tab = match self.active_tab {
            Tab::Generations => Tab::Matchups,
            Tab::Matchups => Tab::Generations,
        };
    }
}
