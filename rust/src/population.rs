use crate::{
    config::EvolutionConfig, constants::*, gamestate::GameState, traits::Individual,
};
use rand::prelude::*;
use rayon::prelude::*;
use std::{
    array,
    sync::atomic::{AtomicU32, Ordering},
};

/// Represents a population of individuals (neural networks) for evolutionary training.
///
/// # Memory Layout
/// - `individuals`: Fixed-size array allocated on the stack (if `I` is stack-allocated, e.g., StackIndividual),
///   otherwise heap for large types (e.g., HeapIndividual).
/// - `fitness`: Array of atomic integers, allowing concurrent fitness updates in parallel engines.
/// - `config`: Evolutionary parameters (mutation rate, elite count, etc.).
///
/// # Algorithm
/// Each generation:
/// 1. Evaluate all individuals (tournament, round-robin)
/// 2. Select elites (top performers)
/// 3. Fill next generation with crossovers and mutations
/// 4. Repeat for a fixed number of generations
pub struct Population<I: Individual> {
    /// The population of neural networks (genomes).
    pub individuals: [I; POPULATION_SIZE],
    /// Fitness scores for each individual (updated in parallel-safe way).
    fitness: [AtomicU32; POPULATION_SIZE],
    /// Evolutionary parameters.
    config: EvolutionConfig,
}

impl<I: Individual> Population<I> {
    /// Creates a new population based on the provided configuration.
    ///
    /// # Parameters
    /// - `config`: Evolutionary parameters (population size, mutation rate, etc.)
    ///
    /// # Returns
    /// New population with randomly initialized individuals and zeroed fitness.
    pub fn new(config: EvolutionConfig) -> Self {
        Self {
            individuals: array::from_fn(|_| I::default()),
            fitness: array::from_fn(|_| AtomicU32::new(0)),
            config,
        }
    }

    /// Evaluates the fitness of all individuals by pitting them against each other
    /// in a round-robin tournament.
    ///
    /// # Algorithm
    /// For every unique pair of individuals, simulates a game and updates their fitness.
    /// Fitness is the number of successful returns (paddle-ball hits).
    /// This is O(n^2) and can be parallelized.
    fn evaluate_fitness(&mut self) {
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }
        let mut game_state = GameState::new();

        for i in 0..POPULATION_SIZE {
            for j in (i + 1)..POPULATION_SIZE {
                let (returns_i, returns_j) =
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
                self.fitness[i].fetch_add(returns_i, Ordering::Relaxed);
                self.fitness[j].fetch_add(returns_j, Ordering::Relaxed);
            }
        }
    }

    /// Parallel version of fitness evaluation using Rayon.
    ///
    /// # Memory/Threading
    /// - Each thread gets its own `GameState` to avoid mutation races.
    /// - Fitness is updated atomically.
    fn evaluate_fitness_concurrent(&mut self) {
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }

        let pairs: Vec<(usize, usize)> = (0..POPULATION_SIZE)
            .flat_map(|i| (i + 1..POPULATION_SIZE).map(move |j| (i, j)))
            .collect();

        pairs.par_iter().for_each(|&(i, j)| {
            // Each thread needs its own GameState
            let mut game_state = GameState::new();
            let (returns_i, returns_j) =
                game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
            self.fitness[i].fetch_add(returns_i, Ordering::Relaxed);
            self.fitness[j].fetch_add(returns_j, Ordering::Relaxed);
        });
    }

    /// Selects the fittest individuals and returns an array of their indices sorted by fitness.
    ///
    /// # Algorithm
    /// Sorts indices by fitness descending. Elites are the first `elite_count` indices.
    ///
    /// # Parameters
    /// - `elite_count` (from config): Controls how many top individuals survive unchanged.
    fn select_elites(&self) -> [usize; POPULATION_SIZE] {
        let mut indices: [usize; POPULATION_SIZE] = array::from_fn(|i| i);
        // Sort by fitness in descending order.
        indices.sort_unstable_by(|&a, &b| {
            self.fitness[b]
                .load(Ordering::Relaxed)
                .cmp(&self.fitness[a].load(Ordering::Relaxed))
        });
        indices
    }

    /// Fills the next generation with offspring from elites, using crossover and mutation.
    ///
    /// # Algorithm
    /// - For each non-elite individual:
    ///   1. Select two random parents from elites.
    ///   2. Perform crossover to create a child genome.
    ///   3. Mutate the child genome (with probability/magnitude from config).
    ///   4. Replace the non-elite with the new child.
    ///
    /// # Parameters
    /// - `mutation_rate`/`mutation_strength`: Control how much and how often genes are mutated.
    /// - `elite_count`: Number of survivors per generation.
    fn recombination_and_mutation(&mut self, sorted_indices: &[usize; POPULATION_SIZE]) {
        let mut rng = thread_rng();

        // The first `ELITE_COUNT` indices point to the elites who will be parents.
        let elite_parent_indices = &sorted_indices[0..ELITE_COUNT];

        // The remaining indices point to non-elites that will be replaced.
        for i in ELITE_COUNT..POPULATION_SIZE {
            let individual_to_replace_idx = sorted_indices[i];

            // Select two random elite parents.
            let p1_idx = *elite_parent_indices.choose(&mut rng).unwrap();
            let p2_idx = *elite_parent_indices.choose(&mut rng).unwrap();

            let p1 = self.individuals[p1_idx].clone();
            let p2 = self.individuals[p2_idx].clone();

            // Recombine and mutate in-place.
            self.individuals[individual_to_replace_idx].recombine_from(
                &p1,
                &p2,
                &mut rng,
                &self.config,
            );
        }
    }

    /// Runs the complete evolutionary process for a specified number of generations.
    ///
    /// # Algorithm
    /// 1. For each generation:
    ///    - Evaluate all individuals (fitness)
    ///    - Select elites
    ///    - Fill next generation with crossovers/mutations
    ///    - Print progress statistics
    /// 2. After final generation, save the best genome to disk.
    ///
    /// # Parameters
    /// - `generations`: Number of generations to run
    /// - `concurrent`: If true, enables parallel fitness evaluation
    pub fn evolve(&mut self) {
        println!("Starting evolution...");
        for gen in 0..self.config.generations {
            if self.config.concurrent {
                self.evaluate_fitness_concurrent();
            } else {
                self.evaluate_fitness();
            }

            let sorted_indices = self.select_elites();

            let best_fitness = self.fitness[sorted_indices[0]].load(Ordering::Relaxed);
            let worst_fitness = self.fitness[sorted_indices[POPULATION_SIZE - 1]].load(Ordering::Relaxed);
            let average_fitness = self.fitness.iter().map(|f| f.load(Ordering::Relaxed)).sum::<u32>()
                as f32
                / POPULATION_SIZE as f32;

            println!(
                "Gen {:<3} | Best: {:<5} | Avg: {:<7.2} | Worst: {}",
                gen,
                best_fitness,
                average_fitness,
                worst_fitness
            );

            self.recombination_and_mutation(&sorted_indices);
        }
        // After evolution, find the best individual and save it.
        let sorted_indices = self.select_elites();
        let best_individual = self.individuals[sorted_indices[0]].clone();
        let file_name = format!("best_{}.net", I::name());
        if let Err(e) = best_individual.save(&file_name) {
            println!("Error saving best network: {}", e);
        } else {
            println!("Best network saved to {}", file_name);
        }

        println!("Evolution finished.");
    }
}
