use std::cmp::Ordering;

use rand::{self, Rng, seq::IteratorRandom};
use rand_distr::Distribution;
use rayon::prelude::*;

use crate::game::state::{Game, Side};

const BIAS_INDICES: [usize; 21] = {
    let mut arr = [0; 21];
    let mut idx = 0;
    let mut i = 128;
    while i <= 143 {
        arr[idx] = i;
        idx += 1;
        i += 1;
    }
    i = 208;
    while i <= 211 {
        arr[idx] = i;
        idx += 1;
        i += 1;
    }
    arr[20] = 216;
    arr
};

const WEIGHTS: usize = 9 * 16 + 17 * 4 + 5 * 1;

const MEAN: f32 = 0.;
const STD_DEV: f32 = 0.1;

/// An individual neural network in the population.
/// Contains its weights and fitness score from tournament evaluation.
pub struct Individual {
    pub weights: [f32; WEIGHTS],
    pub fitness: i8,
}

impl Default for Individual {
    fn default() -> Self {
        let mut array = [0.; WEIGHTS];
        // Initialize weights with random values in parallel
        array.par_iter_mut().enumerate().for_each(|(i, w)| {
            if !BIAS_INDICES.contains(&i) {
                *w = 1.;
            } else {
                *w = rand::rng().random_range(-1. ..=1.);
            }
        });
        Self {
            weights: array,
            fitness: 0,
        }
    }
}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl Eq for Individual {}

impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fitness.partial_cmp(&other.fitness)
    }
}

impl Ord for Individual {
    fn cmp(&self, other: &Self) -> Ordering {
        self.fitness.cmp(&other.fitness)
    }
}

impl Individual {
    /// Mutates all weights by adding a small random value from a normal distribution and resets fitness.
    fn mutate(&mut self) {
        self.weights
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, weight)| {
                if !BIAS_INDICES.contains(&i) {
                    return;
                }
                *weight += rand_distr::Normal::new(MEAN, STD_DEV)
                    .unwrap()
                    .sample(&mut rand::rng()) as f32;
                *weight = (*weight).clamp(-1., 1.);
            });
        self.fitness = 0;
    }
}

/// A population of neural networks undergoing evolutionary training.
/// Designed as immutable to maintain a single memory structure without copying.
pub struct Group {
    pub individuals: Vec<Individual>,
}

impl Group {
    /// Creates a new population with 256 randomly initialized individuals.
    pub fn new(pop_size: usize) -> Self {
        Self {
            individuals: (0..pop_size)
                .into_par_iter()
                .map(|_| Individual::default())
                .collect(),
        }
    }

    /// Mutates all individuals in the population in parallel.
    pub fn mutate(&mut self, elites: usize) {
        self.individuals
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, individual)| {
                if i >= elites {
                    individual.mutate();
                }
            });
    }

    /// Performs tournament evaluation on all individuals.
    /// Each individual fights against 4 randomly chosen opponents.
    pub fn train(&mut self, game: &mut Game, tournament_size: usize) {
        let len = self.individuals.len();
        for i in 0..len {
            // Split array to get current individual and all others
            let (before, rest) = self.individuals.split_at_mut(i);
            let (current, after) = rest.split_first_mut().unwrap();

            // Select random opponents from remaining individuals
            let others: Vec<&Individual> = before
                .iter()
                .chain(after.iter())
                .choose_multiple(&mut rand::rng(), tournament_size - 1);

            // Fight against each opponent
            for (i, &opponent) in others.iter().enumerate() {
                match game.run_until(current, opponent) {
                    Side::Left => current.fitness += 1,
                    Side::Right => current.fitness -= 1,
                }
            }
        }
    }
}
