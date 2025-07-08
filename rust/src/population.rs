use crate::{
    config::Config,
    gamestate::GameState,
    traits::Individual,
    tui::training::TrainingMessage,
};
use rand::{prelude::*, rng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use tracing::{debug, info, trace};

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
    /// Fitness scores for each individual. A `u64` packs two `u32` scores:
    /// - The higher 32 bits are the primary score (e.g., returns + shots).
    /// - The lower 32 bits are the secondary score (e.g., wins).
    /// This allows for atomic updates and a simple multi-objective sort.
    pub fitness: Vec<AtomicU64>,
    /// Evolutionary parameters.
    pub config: Config,
}

impl<I: Individual> Population<I> {
    /// Creates a new population based on the provided configuration.
    ///
    /// # Parameters
    /// - `config`: Evolutionary parameters, including `population_size`.
    ///
    /// # Returns
    /// A new population with randomly initialized individuals and zeroed fitness scores.
    pub fn new(config: Config) -> Self {
        let pop_size = config.population_size;
        Self {
            individuals: (0..pop_size).map(|_| I::default()).collect(),
            fitness: (0..pop_size).map(|_| AtomicU64::new(0)).collect(),
            config,
        }
    }

    /// Evaluates the fitness of all individuals by pitting them against each other
    /// in a round-robin tournament, sending real-time progress updates.
    pub fn evaluate_fitness(&mut self, _tx: &Option<mpsc::Sender<TrainingMessage>>) {
        trace!("Resetting fitness scores.");
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }

        let pop_size = self.config.population_size;
        trace!("Starting sequential full tournament (C++ equivalent).");

        for i in 0..pop_size {
            for j in 0..pop_size {
                if i == j { continue; } // Don't play against yourself
                
                // Run simulation
                let mut game_state = GameState::new();
                let ((left_primary, left_secondary), (right_primary, right_secondary)) =
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);

                // Update fitness
                let left_packed_score = ((left_primary as u64) << 32) | (left_secondary as u64);
                let right_packed_score = ((right_primary as u64) << 32) | (right_secondary as u64);
                self.fitness[i].fetch_add(left_packed_score, Ordering::Relaxed);
                self.fitness[j].fetch_add(right_packed_score, Ordering::Relaxed);
            }
        }
        trace!("Sequential full tournament finished.");
    }

    /// A parallel version of `evaluate_fitness` using Rayon, sending real-time progress.
    pub fn evaluate_fitness_concurrent(&mut self, tx: &Option<mpsc::Sender<TrainingMessage>>) {
        trace!("Resetting fitness scores.");
        self.fitness
            .par_iter()
            .for_each(|f| f.store(0, Ordering::Relaxed));

        let pop_size = self.config.population_size;
        let pairs: Vec<(usize, usize)> = (0..pop_size)
            .flat_map(|i| (0..pop_size).filter(move |&j| i != j).map(move |j| (i, j)))
            .collect();

        trace!(
            total_games = pairs.len(),
            "Starting concurrent full tournament (C++ equivalent)."
        );

        if let Some(tx) = tx {
            // We have a sender, use for_each_with to clone it for each thread.
            pairs.par_iter().for_each_with(
                tx.clone(),
                |_thread_tx, &(i, j)| {
                    let mut game_state = GameState::new();
                    let ((left_primary, left_secondary), (right_primary, right_secondary)) =
                        game_state.simulate(
                            &self.individuals[i],
                            &self.individuals[j],
                            &self.config,
                        );

                    // Pack scores into u64 and update atomically
                    let left_packed_score = ((left_primary as u64) << 32) | (left_secondary as u64);
                    let right_packed_score =
                        ((right_primary as u64) << 32) | (right_secondary as u64);

                    self.fitness[i].fetch_add(left_packed_score, Ordering::Relaxed);
                    self.fitness[j].fetch_add(right_packed_score, Ordering::Relaxed);
                },
            );
        } else {
            // No sender, just run the simulations.
            pairs.par_iter().for_each(|&(i, j)| {
                let mut game_state = GameState::new();
                let ((left_primary, left_secondary), (right_primary, right_secondary)) =
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);

                // Pack scores into u64 and update atomically
                let left_packed_score = ((left_primary as u64) << 32) | (left_secondary as u64);
                let right_packed_score = ((right_primary as u64) << 32) | (right_secondary as u64);

                self.fitness[i].fetch_add(left_packed_score, Ordering::Relaxed);
                self.fitness[j].fetch_add(right_packed_score, Ordering::Relaxed);
            });
        }
        trace!("Concurrent full tournament finished.");
    }

    /// Selects the fittest individuals and returns their indices, sorted by fitness.
    pub fn select_elites(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.config.population_size).collect();
        // Sort by the packed u64 fitness score. Since the primary score is in the
        // most significant bits, this automatically handles the multi-objective sort.
        indices.sort_by_key(|&i| self.fitness[i].load(Ordering::Relaxed));
        indices.reverse(); // Higher scores are better
        indices
    }

    /// Fills the next generation with offspring from elites, using a selectable
    /// reproduction strategy.
    fn recombination_and_mutation(&mut self, sorted_indices: &[usize]) {
        let mut next_generation = Vec::with_capacity(self.config.population_size);
        let mut rng = rng();

        match self.config.reproduction_strategy {
            crate::config::ReproductionStrategy::CppEquivalent => {
                // C++-equivalent reproduction strategy.
                let survivor_count = (self.config.population_size as f32).sqrt() as usize;

                // 1. Elitism: Carry over survivors.
                for i in 0..survivor_count {
                    next_generation.push(self.individuals[sorted_indices[i]].clone());
                }

                // 2. Crossover: All pairs of survivors.
                let mut current_member_idx = survivor_count;
                'crossover: for i in 0..survivor_count {
                    for j in (i + 1)..survivor_count {
                        if current_member_idx >= self.config.population_size {
                            break 'crossover;
                        }
                        let parent1 = &self.individuals[sorted_indices[i]];
                        let parent2 = &self.individuals[sorted_indices[j]];
                        let offspring = parent1.crossover(parent2, &mut rng);
                        next_generation.push(offspring);
                        current_member_idx += 1;
                    }
                }

                // 3. Mutation: Fill the rest with mutated survivors.
                while current_member_idx < self.config.population_size {
                    for i in 0..survivor_count {
                        if current_member_idx >= self.config.population_size {
                            break;
                        }
                        let mut offspring = self.individuals[sorted_indices[i]].clone();
                        offspring.mutate(&mut rng, &self.config);
                        next_generation.push(offspring);
                        current_member_idx += 1;
                    }
                }
            }
            crate::config::ReproductionStrategy::Modern => {
                // Modern Rust-native reproduction strategy.
                let elite_count = self.config.elite_count;

                // 1. Elitism: Carry over the best individuals directly.
                for i in 0..elite_count {
                    next_generation.push(self.individuals[sorted_indices[i]].clone());
                }

                // 2. Fill the rest with children from crossover and mutation of elites.
                for _ in elite_count..self.config.population_size {
                    let parent1_idx = sorted_indices[rng.random_range(0..elite_count)];
                    let parent2_idx = sorted_indices[rng.random_range(0..elite_count)];
                    let mut offspring = self.individuals[parent1_idx]
                        .crossover(&self.individuals[parent2_idx], &mut rng);
                    offspring.mutate(&mut rng, &self.config);
                    next_generation.push(offspring);
                }
            }
        }

        // Ensure the population size is correct.
        next_generation.truncate(self.config.population_size);
        self.individuals = next_generation;
    }

    /// Runs the complete evolutionary process for a specified number of generations,
    /// sending progress updates to the UI thread via a channel.
    ///
    /// # Parameters
    /// - `tx`: The sending end of a channel to communicate with the UI thread.
    ///
    /// # Algorithm
    /// This is the main loop of the genetic algorithm:
    /// 1. For each generation:
    ///    a. Signal the start of the generation to the UI.
    ///    b. Evaluate all individuals, sending real-time matchup updates.
    ///    c. Select the elites by sorting individuals based on their fitness scores.
    ///    d. Send a summary `Progress` message to the UI.
    ///    e. Fill the next generation with offspring via crossover and mutation.
    /// 2. After all generations are complete, return the best-performing individual found.
    ///
    /// # Teaching Note: Decoupling with Message Passing
    /// This `evolve` function has been refactored from a callback-based approach to a
    /// message-passing one. Instead of invoking a function provided by the caller, it sends
    /// structured `TrainingMessage` enums over a channel. This is a more robust and flexible
    /// pattern for concurrent applications. It decouples the core evolution logic from the
    /// UI, allowing either to be changed independently. If the UI thread closes, the `tx.send`
    /// operations will fail, allowing the evolution to terminate gracefully.
    pub fn evolve(&mut self, tx: Option<mpsc::Sender<TrainingMessage>>) -> I {
        debug!(
            generations = self.config.generations,
            "Starting evolution loop."
        );
        for gen in 0..self.config.generations {
            debug!(generation = gen + 1, "Starting generation.");



            trace!("Evaluating fitness...");
            if self.config.concurrent {
                self.evaluate_fitness_concurrent(&tx);
            } else {
                self.evaluate_fitness(&tx);
            }
            trace!("Fitness evaluation complete.");

            trace!("Selecting elites...");
            let sorted_indices = self.select_elites();
            trace!("Elite selection complete.");

            let best_fitness_packed = self.fitness[sorted_indices[0]].load(Ordering::Relaxed);
            let worst_fitness_packed = self.fitness
                [sorted_indices[self.config.population_size - 1]]
                .load(Ordering::Relaxed);

            // Unpack the primary score for reporting
            let best_fitness = (best_fitness_packed >> 32) as f32;
            let worst_fitness = (worst_fitness_packed >> 32) as f32;

            let average_fitness = self
                .fitness
                .iter()
                .map(|f| (f.load(Ordering::Relaxed) >> 32) as u32) // Use primary score for average
                .sum::<u32>() as f32
                / self.config.population_size as f32;

            let best_individual = &self.individuals[sorted_indices[0]];

            // Send the end-of-generation progress summary.
            let progress_message = TrainingMessage::Progress {
                generation: (gen + 1) as usize,
                best_fitness: best_fitness as f32,
                avg_fitness: average_fitness,
                worst_fitness: worst_fitness as f32,
                genome_weights: best_individual.weights_as_slice().to_vec(),
            };

            if let Some(tx) = &tx {
                if tx.send(progress_message).is_err() {
                    // Early exit requested by the caller. Return the best individual found so far.
                    debug!(
                        generation = gen + 1,
                        "Evolution stopped early: UI channel closed."
                    );
                    return self.individuals[sorted_indices[0]].clone();
                }
            } else {
                // CLI mode: print progress to stdout.
                info!(
                    "Gen: {:3} | Best Fitness: {:7.2} | Avg Fitness: {:7.2} | Worst Fitness: {:7.2}",
                    gen + 1,
                    best_fitness,
                    average_fitness,
                    worst_fitness
                );
            }

            trace!("Performing recombination and mutation...");
            self.recombination_and_mutation(&sorted_indices);
            trace!("Recombination and mutation complete.");
            debug!(generation = gen + 1, "Generation complete.");
        }

        // After evolution, find and return the best individual.
        debug!("Evolution loop finished. Selecting final best individual.");
        let sorted_indices = self.select_elites();
        self.individuals[sorted_indices[0]].clone()
    }
}
