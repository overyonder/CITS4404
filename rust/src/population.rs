use crate::{constants::*, gamestate::GameState, individual::Individual};
use rand::prelude::*;
use rand::thread_rng;
use rand_distr::Normal;
use std::array;

pub struct Population {
    pub individuals: [Individual; POPULATION_SIZE],
    fitness: [u32; POPULATION_SIZE],
}

impl Population {
    /// Creates the initial population with random individuals and zero fitness.
    pub fn abiogenesis() -> Self {
        Self {
            individuals: array::from_fn(|_| Individual::default()),
            fitness: [0; POPULATION_SIZE],
        }
    }

    /// Evaluates the fitness of all individuals by pitting them against each other
    /// in a round-robin tournament.
    fn evaluate_fitness(&mut self) {
        self.fitness.fill(0);
        let mut game_state = GameState::new();

        for i in 0..POPULATION_SIZE {
            for j in (i + 1)..POPULATION_SIZE {
                let (returns_i, returns_j) = game_state.simulate(&self.individuals[i], &self.individuals[j]);
                self.fitness[i] += returns_i;
                self.fitness[j] += returns_j;
            }
        }
    }

    /// Selects the fittest individuals and returns an array of their indices sorted by fitness.
    fn select_elites(&self) -> [usize; POPULATION_SIZE] {
        let mut indices: [usize; POPULATION_SIZE] = array::from_fn(|i| i);
        // Sort by fitness in descending order.
        indices.sort_unstable_by(|&a, &b| self.fitness[b].cmp(&self.fitness[a]));
        indices
    }

    /// Replaces non-elite individuals with offspring from the elites using crossover,
    /// then applies mutation to the new offspring. This is done in-place.
    fn recombination_and_mutation(&mut self, sorted_indices: &[usize; POPULATION_SIZE]) {
        let mut rng = thread_rng();
        let normal_dist = Normal::new(0.0, MUTATION_STRENGTH).unwrap();

        // The first `ELITE_COUNT` indices point to the elites who will be parents.
        let elite_parent_indices = &sorted_indices[0..ELITE_COUNT];

        // The remaining indices point to non-elites that will be replaced.
        for i in ELITE_COUNT..POPULATION_SIZE {
            let individual_to_replace_idx = sorted_indices[i];

            // --- Crossover ---
            // Select two random elite parents.
            let p1_idx = *elite_parent_indices.choose(&mut rng).unwrap();
            let p2_idx = *elite_parent_indices.choose(&mut rng).unwrap();

            // Copy parent weights to the stack to satisfy the borrow checker for in-place mutation.
            // This is a small, fast copy.
            let p1_weights = self.individuals[p1_idx].weights;
            let p2_weights = self.individuals[p2_idx].weights;

            let offspring = &mut self.individuals[individual_to_replace_idx];
            for j in 0..TOTAL_WEIGHTS {
                offspring.weights[j] = if rng.gen() { p1_weights[j] } else { p2_weights[j] };
            }

            // --- Mutation ---
            for j in 0..TOTAL_WEIGHTS {
                if rng.gen::<f32>() < MUTATION_RATE {
                    offspring.weights[j] += normal_dist.sample(&mut rng);
                }
            }
        }
    }

    /// Runs the complete evolutionary process for a specified number of generations.
    pub fn evolve(&mut self, generations: u32) {
        println!("Starting evolution...");
        for gen in 0..generations {
            self.evaluate_fitness();

            let sorted_indices = self.select_elites();

            let best_fitness = self.fitness[sorted_indices[0]];
            let worst_fitness = self.fitness[sorted_indices[POPULATION_SIZE - 1]];
            let average_fitness = self.fitness.iter().sum::<u32>() as f32 / POPULATION_SIZE as f32;

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
        let best_individual = self.individuals[sorted_indices[0]];
        if let Err(e) = best_individual.save(BEST_NET_FILE) {
            println!("Error saving best network: {}", e);
        } else {
            println!("Best network saved to {}", BEST_NET_FILE);
        }

        println!("Evolution finished.");
    }
}
