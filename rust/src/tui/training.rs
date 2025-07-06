//! Training state, message passing, and backend logic for evolutionary training in the TUI.
use crate::config::Config;

/// Holds the state for the training view.
#[allow(dead_code)] // Struct is used, but some fields are not yet displayed in the UI.
/// Represents the state of a single matchup in the tournament grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GenerationState {
    Pending,
    InProgress,
    Completed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatchupState {
    Pending,
    InProgress,
    Completed,
}

pub struct TrainingState {
    pub running: bool,
    pub current_generation: usize,
    pub total_generations: usize,
    pub best_fitness: f32,
    pub fitness_history: Vec<f32>,
    pub genome_weights: Vec<f32>,
    pub matchups: Vec<MatchupState>,
    pub generations: Vec<GenerationState>,
}

impl TrainingState {
    pub fn new(config: &Config) -> Self {
        let pop_size = config.population_size;
        let total_matchups = if pop_size > 1 {
            pop_size * (pop_size - 1) / 2
        } else {
            0
        };
        Self {
            running: true,
            current_generation: 0,
            total_generations: config.generations as usize,
            best_fitness: 0.0,
            fitness_history: Vec::new(),
            genome_weights: Vec::new(),
            matchups: vec![MatchupState::Pending; total_matchups],
            generations: vec![GenerationState::Pending; config.generations as usize],
        }
    }
}

/// Messages sent from the training thread to the UI thread.
#[allow(dead_code)] // The UI doesn't handle all message types or fields yet.
pub enum TrainingMessage {
    /// Signals the start of a new generation.
    GenerationStart { total_matchups: usize },
    /// Updates the state of a single matchup in the grid.
    MatchupUpdate {
        matchup_index: usize,
        state: MatchupState,
    },
    /// Reports summary progress at the end of a generation.
    Progress {
        generation: usize,
        best_fitness: f32,
        avg_fitness: f32,
        worst_fitness: f32,
        genome_weights: Vec<f32>,
    },
    /// Indicates that the training process has finished.
    Finished,
}
