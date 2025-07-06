use crate::{
    config::Config,
    gamestate::GameState,
    traits::Individual,
    tui::training::{MatchupState, TrainingMessage},
};
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
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
    /// Fitness scores for each individual (updated in a parallel-safe way).
    pub fitness: Vec<AtomicU32>,
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
            fitness: (0..pop_size).map(|_| AtomicU32::new(0)).collect(),
            config,
        }
    }

    /// Calculates a unique index for a matchup between individuals `i` and `j`.
    /// This is used for the "defragmenter"-style UI grid.
    ///
    /// # Formula
    /// The formula treats the matchups as a lower triangular matrix (excluding the diagonal)
    /// and calculates the flat index for the `(i, j)` pair.
    fn get_matchup_index(&self, i: usize, j: usize) -> usize {
        // Ensure i > j for a consistent index, matching the loop structure.
        let (p1, p2) = if i > j { (i, j) } else { (j, i) };

        // This formula calculates the index for a matchup in a round-robin tournament
        // for the lower triangle of the matchup matrix (where i > j).
        // It's based on the sum of an arithmetic series.
        // For a given `p1`, it has played `p1` games against `0..p1-1`.
        // The number of games for all individuals before `p1` is the sum 0+1+2+...+(p1-1).
        let previous_rows_games = p1 * (p1 - 1) / 2;
        previous_rows_games + p2
    }

    /// Evaluates the fitness of all individuals by pitting them against each other
    /// in a round-robin tournament, sending real-time progress updates.
    pub fn evaluate_fitness(&mut self, tx: &Option<mpsc::Sender<TrainingMessage>>) {
        trace!("Resetting fitness scores.");
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }
        let mut game_state = GameState::new();

        let total_games = self.config.population_size * (self.config.population_size - 1) / 2;
        trace!(total_games, "Starting sequential round-robin tournament.");

        for i in 0..self.config.population_size {
            for j in (i + 1)..self.config.population_size {
                let matchup_index = self.get_matchup_index(i, j);

                if let Some(tx) = tx {
                    // Send InProgress update. Ignore error, as we'll catch it in the main loop.
                    let _ = tx.send(TrainingMessage::MatchupUpdate {
                        matchup_index,
                        state: MatchupState::InProgress,
                    });
                }

                game_state.reset();
                game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
                let (p1_score, p2_score) = game_state.scores;

                if p1_score > p2_score {
                    self.fitness[i].fetch_add(1, Ordering::Relaxed);
                } else if p2_score > p1_score {
                    self.fitness[j].fetch_add(1, Ordering::Relaxed);
                }

                if let Some(tx) = tx {
                    // Send Completed update
                    let _ = tx.send(TrainingMessage::MatchupUpdate {
                        matchup_index,
                        state: MatchupState::Completed,
                    });
                }
            }
        }
    }

    /// A parallel version of `evaluate_fitness` using Rayon, sending real-time progress.
    pub fn evaluate_fitness_concurrent(&mut self, tx: &Option<mpsc::Sender<TrainingMessage>>) {
        trace!("Resetting fitness scores.");
        self.fitness
            .par_iter()
            .for_each(|f| f.store(0, Ordering::Relaxed));

        let pop_size = self.config.population_size;
        let pairs: Vec<(usize, usize)> = (0..pop_size)
            .flat_map(|i| (i + 1..pop_size).map(move |j| (i, j)))
            .collect();

        trace!(
            total_games = pairs.len(),
            "Starting concurrent round-robin tournament."
        );

        if let Some(tx) = tx {
            // We have a sender, use for_each_with to clone it for each thread.
            pairs
                .par_iter()
                .for_each_with(tx.clone(), |thread_tx, &(i, j)| {
                    let matchup_index = self.get_matchup_index(i, j);

                    let _ = thread_tx.send(TrainingMessage::MatchupUpdate {
                        matchup_index,
                        state: MatchupState::InProgress,
                    });

                    let mut game_state = GameState::new();
                    game_state.reset();
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
                    let (p1_score, p2_score) = game_state.scores;

                    if p1_score > p2_score {
                        self.fitness[i].fetch_add(1, Ordering::Relaxed);
                    } else if p2_score > p1_score {
                        self.fitness[j].fetch_add(1, Ordering::Relaxed);
                    }

                    let _ = thread_tx.send(TrainingMessage::MatchupUpdate {
                        matchup_index,
                        state: MatchupState::Completed,
                    });
                });
        } else {
            // No sender, just run the simulations.
            pairs.par_iter().for_each(|&(i, j)| {
                let mut game_state = GameState::new();
                game_state.reset();
                game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
                let (p1_score, p2_score) = game_state.scores;

                if p1_score > p2_score {
                    self.fitness[i].fetch_add(1, Ordering::Relaxed);
                } else if p2_score > p1_score {
                    self.fitness[j].fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        trace!("Concurrent round-robin tournament finished.");
    }

    /// Selects the fittest individuals and returns their indices, sorted by fitness.
    pub fn select_elites(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.config.population_size).collect();
        indices.sort_by_key(|&i| std::cmp::Reverse(self.fitness[i].load(Ordering::Relaxed)));
        indices
    }

    /// Fills the next generation with offspring from elites, using crossover and mutation.
    fn recombination_and_mutation(&mut self, sorted_indices: &[usize]) {
        let mut rng = rand::rng();
        let elite_count = self.config.elite_count;
        let population_size = self.config.population_size;

        let elite_parent_indices = &sorted_indices[0..elite_count];

        for i in elite_count..population_size {
            let individual_to_replace_idx = sorted_indices[i];

            let p1_idx = *elite_parent_indices.choose(&mut rng).unwrap();
            let p2_idx = *elite_parent_indices.choose(&mut rng).unwrap();

            let p1 = &self.individuals[p1_idx];
            let p2 = &self.individuals[p2_idx];

            let mut child = p1.crossover(p2, &mut rng);
            child.mutate(&mut rng, &self.config);

            self.individuals[individual_to_replace_idx] = child;
        }
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

            let pop_size = self.config.population_size;
            let total_matchups = if pop_size > 1 {
                pop_size * (pop_size - 1) / 2
            } else {
                0
            };

            // Signal the start of a new generation.
            if let Some(tx) = &tx {
                if tx
                    .send(TrainingMessage::GenerationStart { total_matchups })
                    .is_err()
                {
                    debug!("Evolution stopped early: UI channel closed.");
                    let sorted_indices = self.select_elites();
                    return self.individuals[sorted_indices[0]].clone();
                }
            }

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

            let best_fitness = self.fitness[sorted_indices[0]].load(Ordering::Relaxed);
            let worst_fitness = self.fitness[sorted_indices[self.config.population_size - 1]]
                .load(Ordering::Relaxed);
            let average_fitness = self
                .fitness
                .iter()
                .map(|f| f.load(Ordering::Relaxed))
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
