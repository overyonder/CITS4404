use rand::{self, seq::IteratorRandom, Rng, distributions::Normal};
use rayon::prelude::*;

const WEIGHTS: usize = 9 * 16 + 17 * 4 + 5 * 1;
const TOURNAMENT_SIZE: usize = 5;

const MEAN: f32 = 0.0;
const STD_DEV: f32 = 1.0; // A common default standard deviation is 1.0

/// An individual neural network in the population.
/// Contains its weights and fitness score from tournament evaluation.
pub struct Individual {
    weights: [f32; WEIGHTS],
    fitness: i8,
}

impl Default for Individual {
    fn default() -> Self {
        let mut array = [0.0; WEIGHTS];
        // Initialize weights with random values in parallel
        array
            .par_iter_mut()
            .for_each(|w| *w = rand::rng().random());
        Self { weights: array, fitness: 0 }
    }
}

impl Individual {
    /// Mutates all weights by adding a small random value from a normal distribution and resets fitness.
    fn mutate(&mut self) {
        self.weights
            .par_iter_mut()
            .for_each(|weight| {
                let normal = rand::distributions::Normal::new(MEAN, STD_DEV);
                let delta = normal.sample(rand::rng());
                *weight += delta as f32;
            });
        self.fitness = 0;
    }

    /// Simulates a competition between two networks.
    /// Updates own fitness (+1 for win, -1 for loss).
    /// Note: Does not update opponent fitness as opponents are randomly chosen.
    fn fight(&mut self, challenger: &Individual) {
        // TODO: Implement competition logic
    }
}

/// A population of neural networks undergoing evolutionary training.
/// Designed as immutable to maintain a single memory structure without copying.
pub struct Group {
    individuals: Vec<Individual>,
}

impl Group {
    /// Creates a new population with 256 randomly initialized individuals.
    pub fn new() -> Self {
        Self {
            individuals: (0..256)
                .into_par_iter()
                .map(|_| Individual::default())
                .collect(),
        }
    }

    /// Mutates all individuals in the population in parallel.
    pub fn mutate(&mut self) {
        self.individuals.par_iter_mut().for_each(|individual| {
            individual.mutate();
        });
    }

    /// Performs tournament evaluation on all individuals.
    /// Each individual fights against 4 randomly chosen opponents.
    pub fn train(&mut self) {
        let len = self.individuals.len();
        let mut rng = rand::rng();
    
        for i in 0..len {
            // Split array to get current individual and all others
            let (before, rest) = self.individuals.split_at_mut(i);
            let (current, after) = rest.split_first_mut().unwrap();

            // Select random opponents from remaining individuals
            let others: Vec<&Individual> = before
                .iter()
                .chain(after.iter())
                .choose_multiple(&mut rng, TOURNAMENT_SIZE-1);
    
            // Fight against each opponent
            for opponent in &others {
                current.fight(opponent);
            }
        }
    }
}
