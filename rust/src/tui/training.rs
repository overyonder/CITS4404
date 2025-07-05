//! Training state, message passing, and backend logic for evolutionary training in the TUI.
use crate::{
    config::{Engine, EvolutionConfig},
    engines::{GpuIndividual, HeapIndividual, SimdIndividual, StackIndividual},
    population::Population,
    traits::Individual,
};
use std::{sync::atomic::Ordering, sync::mpsc::Sender, thread};

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
}

impl TrainingState {
    pub fn new(config: &EvolutionConfig) -> Self {
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
pub fn evolve_with_progress(config: EvolutionConfig, tx: Sender<TrainingMessage>) {
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
    config: EvolutionConfig,
    tx: Sender<TrainingMessage>,
) {
    let mut pop = Population::<I>::new(config.clone());

    for gen in 0..config.generations {
        if config.concurrent {
            pop.evaluate_fitness_concurrent();
        } else {
            pop.evaluate_fitness();
        }

        let sorted_indices = pop.select_elites();
        let best_fitness = pop.fitness[sorted_indices[0]].load(Ordering::Relaxed);
        let worst_fitness =
            pop.fitness[sorted_indices[config.population_size - 1]].load(Ordering::Relaxed);
        let avg_fitness = pop
            .fitness
            .iter()
            .map(|f| f.load(Ordering::Relaxed))
            .sum::<u32>() as f32
            / config.population_size as f32;

        let best_genome_weights = pop.individuals[sorted_indices[0]]
            .weights_as_slice()
            .to_vec();

        if tx
            .send(TrainingMessage::Progress {
                generation: gen as usize,
                best_fitness: best_fitness as f32,
                avg_fitness,
                worst_fitness: worst_fitness as f32,
                genome_weights: best_genome_weights,
            })
            .is_err()
        {
            // UI thread has likely closed, stop training.
            break;
        }

        pop.recombination_and_mutation(&sorted_indices);
    }

    let _ = tx.send(TrainingMessage::Finished);
}
