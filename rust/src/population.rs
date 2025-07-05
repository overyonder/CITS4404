use crate::{config::EvolutionConfig, gamestate::GameState, traits::Individual};
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Represents a population of individuals (neural networks) for evolutionary training.
///
/// # Fields
/// - `individuals`: A vector containing the population of neural networks (genomes).
/// - `fitness`: A vector of atomic integers for fitness scores, allowing concurrent updates.
/// - `config`: The configuration for the evolutionary process.
///
/// # Algorithm
/// Each generation, the population undergoes:
/// 1. **Evaluation**: All individuals compete in a round-robin tournament to determine fitness.
/// 2. **Selection**: The best-performing individuals (elites) are identified.
/// 3. **Reproduction**: The rest of the population is replaced by offspring created
///    through crossover and mutation of the elites.
/// This cycle repeats for a configured number of generations.
///
/// # Memory
/// - Individuals and fitness scores are stored on the heap in `Vec`s, allowing for
///   a dynamically configurable population size.
///
/// # Teaching Note
/// This struct is a good example of managing a collection of evolving agents.
/// Using a `Vec` instead of a fixed-size array makes the genetic algorithm more
/// flexible, as the population size can be changed at runtime via configuration.
pub struct Population<I: Individual> {
    /// The population of neural networks (genomes).
    pub individuals: Vec<I>,
    /// Fitness scores for each individual (updated in a parallel-safe way).
    pub fitness: Vec<AtomicU32>,
    /// Evolutionary parameters.
    pub config: EvolutionConfig,
}

impl<I: Individual> Population<I> {
    /// Creates a new population based on the provided configuration.
    ///
    /// # Parameters
    /// - `config`: Evolutionary parameters, including `population_size`.
    ///
    /// # Returns
    /// A new population with randomly initialized individuals and zeroed fitness scores.
    pub fn new(config: EvolutionConfig) -> Self {
        let pop_size = config.population_size;
        Self {
            individuals: (0..pop_size).map(|_| I::default()).collect(),
            fitness: (0..pop_size).map(|_| AtomicU32::new(0)).collect(),
            config,
        }
    }

    /// Evaluates the fitness of all individuals by pitting them against each other
    /// in a round-robin tournament.
    ///
    /// # Algorithm
    /// For every unique pair of individuals, a game is simulated. The fitness score
    /// for each individual is the total number of successful returns (paddle-ball hits)
    /// they achieve across all games. This is an O(n^2) operation.
    ///
    /// # Teaching Note
    /// This is a common approach for fitness evaluation in competitive co-evolution.
    /// The fitness of an individual is relative to the performance of others in the
    /// current population.
    pub fn evaluate_fitness(&mut self) {
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }
        let mut game_state = GameState::new();

        for i in 0..self.config.population_size {
            for j in (i + 1)..self.config.population_size {
                let (returns_i, returns_j) =
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
                self.fitness[i].fetch_add(returns_i, Ordering::Relaxed);
                self.fitness[j].fetch_add(returns_j, Ordering::Relaxed);
            }
        }
    }

    /// A parallel version of `evaluate_fitness` using the Rayon crate.
    ///
    /// # Memory/Threading
    /// - The outer loop over individuals is parallelized.
    /// - A new `GameState` is created per thread to prevent data races.
    /// - Fitness scores are updated atomically using `AtomicU32`.
    ///
    /// # Teaching Note
    /// This demonstrates how CPU-bound tasks like fitness evaluation can be easily
    /// parallelized in Rust with Rayon, often providing a significant speedup on
    /// multi-core processors.
    pub fn evaluate_fitness_concurrent(&mut self) {
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }

        (0..self.config.population_size)
            .par_bridge()
            .for_each(|i| {
                let mut game_state = GameState::new();
                for j in (i + 1)..self.config.population_size {
                    let (returns_i, returns_j) = game_state.simulate(
                        &self.individuals[i],
                        &self.individuals[j],
                        &self.config,
                    );
                    self.fitness[i].fetch_add(returns_i, Ordering::Relaxed);
                    self.fitness[j].fetch_add(returns_j, Ordering::Relaxed);
                }
            });
    }

    /// Selects the fittest individuals and returns their indices, sorted by fitness.
    ///
    /// # Returns
    /// A `Vec<usize>` of indices, sorted from highest fitness to lowest.
    ///
    /// # Teaching Note
    /// This is the "selection" phase of a genetic algorithm. The sorted list of
    /// elites is used in the next step to produce the next generation.
    pub fn select_elites(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.config.population_size).collect();
        // Sort by fitness descending. `std::cmp::Reverse` is a convenient wrapper
        // to reverse the sorting order.
        indices.sort_by_key(|&i| std::cmp::Reverse(self.fitness[i].load(Ordering::Relaxed)));
        indices
    }

    /// Fills the next generation with offspring from elites, using crossover and mutation.
    ///
    /// # Algorithm
    /// The top `elite_count` individuals survive unchanged. The remaining individuals
    /// in the population are replaced by new offspring. Each new child is created by:
    ///   1. Selecting two random parents from the elites.
    ///   2. Performing crossover to create a child genome.
    ///   3. Mutating the child's genome based on `mutation_rate` and `mutation_strength`.
    ///
    /// # Teaching Note
    /// This function implements the "crossover" and "mutation" phases. Elitism ensures
    /// that the best solutions are never lost from one generation to the next.
    pub fn recombination_and_mutation(&mut self, sorted_indices: &[usize]) {
        let mut rng = thread_rng();
        let elite_count = self.config.elite_count;
        let population_size = self.config.population_size;

        // The first `elite_count` indices point to the elites who will be parents.
        let elite_parent_indices = &sorted_indices[0..elite_count];

        // The remaining indices point to non-elites that will be replaced.
        for i in elite_count..population_size {
            let individual_to_replace_idx = sorted_indices[i];

            // Select two random elite parents.
            let p1_idx = *elite_parent_indices.choose(&mut rng).unwrap();
            let p2_idx = *elite_parent_indices.choose(&mut rng).unwrap();

            let p1 = &self.individuals[p1_idx];
            let p2 = &self.individuals[p2_idx];

            // Create a new child through crossover and then mutate it.
            let mut child = p1.crossover(p2, &mut rng);
            child.mutate(&mut rng, &self.config);

            // Replace the old individual with the new child.
            self.individuals[individual_to_replace_idx] = child;
        }
    }

    /// Runs the complete evolutionary process for a specified number of generations.
    ///
    /// # Algorithm
    /// This is the main loop of the genetic algorithm:
    /// 1. For each generation:
    ///    - Evaluate all individuals to get their fitness.
    ///    - Select the elites.
    ///    - Fill the next generation with offspring via crossover and mutation.
    ///    - Print progress statistics for the generation.
    /// 2. After the final generation, save the best-performing genome to a file.
    ///
    /// # Teaching Note
    /// This function ties all the pieces of the genetic algorithm together. It shows
    /// the high-level cycle of evaluation, selection, and reproduction.
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
            let worst_fitness =
                self.fitness[sorted_indices[self.config.population_size - 1]].load(Ordering::Relaxed);
            let average_fitness =
                self.fitness.iter().map(|f| f.load(Ordering::Relaxed)).sum::<u32>() as f32
                    / self.config.population_size as f32;

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
