//! Training state, message passing, and backend logic for evolutionary training in the TUI.
use crate::config::Config;
use std::time::Instant;

/// Holds the state for the training view.


pub struct TrainingState {
    pub running: bool,
    pub start_time: Instant,
    pub current_generation: usize,
    pub total_generations: usize,
    pub best_fitness: f32,
    pub fitness_history: Vec<f32>,
    pub genome_weights: Vec<f32>,
}

impl TrainingState {
    /// Creates a new `TrainingState` with default values.
    #[allow(dead_code)]
    pub fn new(config: &Config) -> Self {
        Self {
            running: true,
            start_time: Instant::now(),
            current_generation: 0,
            total_generations: config.generations as usize,
            best_fitness: 0.0,
            fitness_history: Vec::new(),
            genome_weights: Vec::new(),
        }
    }
}

/// Messages sent from the training thread to the UI thread.
#[allow(dead_code)] // The UI doesn't handle all message types or fields yet.
pub enum TrainingMessage {
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
