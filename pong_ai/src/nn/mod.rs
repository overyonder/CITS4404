use std::cmp::Ordering;

use rand::{self, Rng, seq::IteratorRandom};
use rand_distr::Distribution;
use rayon::prelude::*;

use crate::game::state::{Game, Side};

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
        let mut rng = rand::rng();
        use rand_distr::{Distribution, Normal};
        // First hidden layer (16 neurons, 9 inputs each)
        let std1 = (2.0_f32 / 9.0_f32).sqrt();
        let normal1 = Normal::new(0.0, std1).unwrap();
        for i in 0..(16 * 9) {
            array[i] = normal1.sample(&mut rng);
        }
        // Second hidden layer (4 neurons, 17 inputs each)
        let std2 = (2.0_f32 / 17.0_f32).sqrt();
        let normal2 = Normal::new(0.0, std2).unwrap();
        for i in (16 * 9)..(16 * 9 + 4 * 17) {
            array[i] = normal2.sample(&mut rng);
        }
        // Output layer (1 neuron, 5 inputs)
        let std3 = (2.0_f32 / 5.0_f32).sqrt();
        let normal3 = Normal::new(0.0, std3).unwrap();
        for i in (16 * 9 + 4 * 17)..WEIGHTS {
            array[i] = normal3.sample(&mut rng);
        }
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
        self.weights.par_iter_mut().for_each(|weight| {
            *weight += rand_distr::Normal::new(MEAN, STD_DEV)
                .unwrap()
                .sample(&mut rand::rng()) as f32;
            *weight = (*weight).clamp(-1., 1.);
        });
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
    pub fn mutate(&mut self, elites: usize, pop_size: usize) {
        let (non_elites, elites_slice) = self.individuals.split_at_mut(pop_size - elites);
        non_elites.par_iter_mut().for_each(|individual| {
            let elite_idx = rand::rng().random_range(0..elites);
            individual
                .weights
                .copy_from_slice(&elites_slice[elite_idx].weights);
            individual.mutate();
            individual.fitness = 0;
        });
        // Reset fitness for elites
        for elite in elites_slice.iter_mut() {
            elite.fitness = 0;
        }
    }

    /// Performs tournament evaluation on all individuals.
    /// Each individual fights against 4 randomly chosen opponents.
    pub fn train(&mut self, tournament_size: usize) {
        let len = self.individuals.len();
        use crate::game::state::Game;
        use std::cell::RefCell;
        use thread_local::ThreadLocal;
        // Thread-local Game for each thread
        let games = ThreadLocal::new();
        let fitness_deltas: Vec<i8> = (0..len)
            .into_par_iter()
            .map(|i| {
                let (before, rest) = self.individuals.split_at(i);
                let (current, after) = rest.split_first().unwrap();
                // Select random opponents from remaining individuals
                let others: Vec<&Individual> = before
                    .iter()
                    .chain(after.iter())
                    .choose_multiple(&mut rand::rng(), tournament_size - 1);
                // Use the thread's own Game instance
                let game_cell = games.get_or(|| RefCell::new(Game::default()));
                let mut delta = 0;
                for (j, &opponent) in others.iter().enumerate() {
                    let mut game = game_cell.borrow_mut();
                    *game = Game::default(); // Reset game state
                    match game.run_until(
                        current,
                        opponent,
                        if j % 2 == 0 { Side::Left } else { Side::Right },
                    ) {
                        Side::Left => delta += 1,
                        Side::Right => delta -= 1,
                    }
                }
                delta
            })
            .collect();
        // Second pass: apply fitness deltas
        for (ind, delta) in self.individuals.iter_mut().zip(fitness_deltas) {
            ind.fitness += delta;
        }
    }
}
