//! Training state, message passing, and backend logic for evolutionary training in the TUI.
use crate::{
    config::{Config, Engine},
    engines::{GpuIndividual, HeapIndividual, SimdIndividual, StackIndividual},
    population::Population,
    traits::Individual,
};
use std::{sync::mpsc, thread, time::Instant};

/// Holds the state for the training view.
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
    pub start_time: Instant,
}

impl TrainingState {
    pub fn new(config: &Config) -> Self {
        Self {
            running: true,
            current_generation: 0,
            total_generations: config.generations as usize,
            games_completed: 0, // This will be updated via messages
            best_fitness: 0.0,
            fitness_history: Vec::new(),
            engine: config.engine,
            genome_weights: Vec::new(),
            log: vec!["Starting training...".to_string()],
            start_time: Instant::now(),
        }
    }
}

/// Messages sent from the training thread to the UI thread.
pub enum TrainingMessage {
    /// Reports progress of the training.
    Progress {
        generation: usize,
        best_fitness: f32,
        avg_fitness: f32,
        worst_fitness: f32,
        genome_weights: Vec<f32>,
    },
    /// Indicates that the training process has finished.
    Finished,
    /// A log message to display in the UI.
    Log(String),
}

/// Spawns a new thread to run the evolutionary algorithm, sending progress
/// messages back to the main UI thread.
pub fn evolve_with_progress(config: Config, tx: mpsc::Sender<TrainingMessage>) {
    thread::spawn(move || {
        // The `concurrent` flag is handled by the evolution logic, not the engine enum.
        match config.engine {
            Engine::Stack => run_evolution_for_engine::<StackIndividual>(config, tx),
            Engine::Heap => run_evolution_for_engine::<HeapIndividual>(config, tx),
            Engine::Simd => run_evolution_for_engine::<SimdIndividual>(config, tx),
            Engine::Gpu => run_evolution_for_engine::<GpuIndividual>(config, tx),
        }
    });
}

/// Generic function to run the evolution loop for a specific `Individual` type.
fn run_evolution_for_engine<I: Individual + Clone + Send + Sync + 'static>(
    config: Config,
    tx: mpsc::Sender<TrainingMessage>,
) {
    let mut pop = Population::<I>::new(config);

    // Define a callback that sends progress messages back to the UI thread.
    let evolution_callback = |gen: u32,
                              best_fitness: u32,
                              avg_fitness: f32,
                              worst_fitness: u32,
                              genome_weights: &[f32]| {
        let message = TrainingMessage::Progress {
            generation: gen as usize,
            best_fitness: best_fitness as f32,
            avg_fitness,
            worst_fitness: worst_fitness as f32,
            genome_weights: genome_weights.to_vec(),
        };

        // If sending fails, the UI thread has likely closed. We'll stop the evolution
        // by returning `false`.
        tx.send(message).is_ok()
    };

    // Run the evolution. The `evolve` function will call our callback each generation.
    pop.evolve(evolution_callback);

    // Signal that training is finished, whether it completed successfully or was aborted.
    let _ = tx.send(TrainingMessage::Finished);
}
