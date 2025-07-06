use crate::{config::Config, gamestate::GameState, traits::Individual};
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{debug, trace};

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
    /// current population. The round-robin tournament has a time complexity of O(n²),
    /// where n is the population size, as it requires n * (n-1) / 2 games. This can
    /// become a significant bottleneck, which motivates the concurrent version.
    pub fn evaluate_fitness(&mut self) {
        trace!("Resetting fitness scores.");
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }
        let mut game_state = GameState::new();

        let total_games = self.config.population_size * (self.config.population_size - 1) / 2;
        trace!(total_games, "Starting sequential round-robin tournament.");

        for i in 0..self.config.population_size {
            for j in (i + 1)..self.config.population_size {
                trace!(player1 = i, player2 = j, "Simulating game.");
                let (returns_i, returns_j) =
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
                self.fitness[i].fetch_add(returns_i, Ordering::Relaxed);
                self.fitness[j].fetch_add(returns_j, Ordering::Relaxed);
            }
        }
        trace!("Sequential round-robin tournament finished.");
    }

    /// A parallel version of `evaluate_fitness` using the Rayon crate.
    ///
    /// # Memory/Threading
    /// - A list of all unique game pairings is created upfront.
    /// - Rayon's `par_iter` distributes these pairings across available threads.
    /// - A new `GameState` is created per game to prevent data races.
    /// - Fitness scores are updated atomically using `AtomicU32`.
    ///
    /// # Teaching Note: Concurrency with Rayon and Atomics
    /// This function demonstrates a robust pattern for parallelization. A "work list" (the
    /// `pairs` vector) is created upfront, and Rayon's `par_iter` efficiently distributes
    /// these pairings across all available CPU cores.
    ///
    /// To prevent data races when updating fitness scores from multiple threads
    /// simultaneously, we use `AtomicU32`. The `fetch_add` operation ensures that
    /// increments are thread-safe. We use `Ordering::Relaxed` because the order in which
    /// fitness scores are added doesn't matter; we only need the final sum to be correct.
    /// This is the weakest (and often fastest) memory ordering, making it ideal for
    /// counters like this.
    pub fn evaluate_fitness_concurrent(&mut self) {
        trace!("Resetting fitness scores.");
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }

        let pop_size = self.config.population_size;
        let pairs: Vec<(usize, usize)> = (0..pop_size)
            .flat_map(|i| (i + 1..pop_size).map(move |j| (i, j)))
            .collect();

        trace!(
            total_games = pairs.len(),
            "Starting concurrent round-robin tournament."
        );

        pairs.par_iter().for_each(|&(i, j)| {
            // Create a new GameState for each simulation to ensure thread safety.
            let mut game_state = GameState::new();
            let (returns_i, returns_j) =
                game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);
            self.fitness[i].fetch_add(returns_i, Ordering::Relaxed);
            self.fitness[j].fetch_add(returns_j, Ordering::Relaxed);
        });
        trace!("Concurrent round-robin tournament finished.");
    }

    /// Selects the fittest individuals and returns their indices, sorted by fitness.
    ///
    /// # Returns
    /// A `Vec<usize>` of indices, sorted from highest fitness to lowest.
    ///
    /// # Teaching Note: Elitism
    /// This is the "selection" phase of a genetic algorithm. By sorting the individuals
    /// by fitness, we can identify the top performers, or "elites". The concept of
    /// **elitism** involves carrying over the best individuals to the next generation,
    /// either directly or by making them the exclusive parents for new offspring. This
    /// ensures that the best solutions found so far are not lost.
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
    /// 1.  Identify the `elite_count` best individuals from `sorted_indices` to act as parents.
    /// 2.  Iterate through the `population_size - elite_count` worst-performing individuals,
    ///     which are slated for replacement.
    /// 3.  For each individual to be replaced:
    ///     a. Randomly select two parents from the elite pool.
    ///     b. Create a new `child` by performing `crossover` on the two parents.
    ///     c. Apply `mutate` to the new child to introduce genetic diversity.
    ///     d. Replace the old, underperforming individual with the new child.
    ///
    /// # Teaching Note
    /// This function implements the "reproduction" phase. It ensures that the genetic
    /// material from the most successful individuals is passed on, while mutation continues
    /// to explore the solution space.
    fn recombination_and_mutation(&mut self, sorted_indices: &[usize]) {
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

    /// Runs the complete evolutionary process for a specified number of generations,
    /// providing progress updates via a callback.
    ///
    /// # Type Parameters
    /// - `F`: A closure type that takes progress information as arguments.
    ///
    /// # Parameters
    /// - `on_progress`: A callback invoked each generation with stats and the best genome.
    ///
    /// # Algorithm
    /// This is the main loop of the genetic algorithm:
    /// 1. For each generation:
    ///    a. Evaluate all individuals using either the sequential or concurrent method.
    ///    b. Select the elites by sorting individuals based on their fitness scores.
    ///    c. Invoke the `on_progress` callback with the current generation's statistics.
    ///       This decouples the core logic from the UI and allows for early stopping.
    ///    d. Fill the next generation with offspring via crossover and mutation.
    /// 2. After all generations are complete, return the best-performing individual found.
    ///
    /// # Teaching Note: The Callback Pattern
    /// The `on_progress` closure is a powerful **callback pattern**. It allows the caller
    /// (e.g., the TUI or CLI) to monitor the evolution in real-time. By returning a `bool`,
    /// the callback can also signal the `evolve` loop to terminate early, which is useful
    /// for implementing a "stop" button in a user interface.
    pub fn evolve<F>(&mut self, mut on_progress: F) -> I
    where
        F: FnMut(u32, u32, f32, u32, &[f32]) -> bool, // Return bool to continue
    {
        debug!(generations = self.config.generations, "Starting evolution loop.");
        for gen in 0..self.config.generations {
            debug!(generation = gen + 1, "Starting generation.");

            trace!("Evaluating fitness...");
            if self.config.concurrent {
                self.evaluate_fitness_concurrent();
            } else {
                self.evaluate_fitness();
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

            // Invoke the callback. If it returns false, stop the evolution.
            trace!("Invoking on_progress callback...");
            if !on_progress(
                gen + 1,
                best_fitness,
                average_fitness,
                worst_fitness,
                best_individual.weights_as_slice(),
            ) {
                // Early exit requested by the caller. Return the best individual found so far.
                debug!(generation = gen + 1, "Evolution stopped early by callback.");
                return self.individuals[sorted_indices[0]].clone();
            }
            trace!("on_progress callback finished.");

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
