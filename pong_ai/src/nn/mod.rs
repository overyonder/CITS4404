// No threading for wasm
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use thread_local::ThreadLocal;

use crate::game::state::{Game, Side};
use rand::{self, Rng, seq::IteratorRandom};
use rand_distr::Distribution;
use std::cmp::Ordering;

const WEIGHTS: usize = 9 * 16 + 17 * 4 + 5 * 1;

const MEAN: f32 = 0.;
const STD_DEV: f32 = 0.1;

/// An individual neural network in the population.
/// Contains its weights and fitness score from tournament evaluation.
pub struct Individual {
    weights: [f32; WEIGHTS],
    fitness: i8,
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
        self.weights.iter_mut().for_each(|weight: &mut f32| {
            *weight += rand_distr::Normal::new(MEAN, STD_DEV)
                .unwrap()
                .sample(&mut rand::rng()) as f32;
            *weight = (*weight).clamp(-1., 1.);
        });
    }

    pub fn weights(&self) -> &[f32; WEIGHTS] {
        &self.weights
    }

    pub fn fitness(&self) -> &i8 {
        &self.fitness
    }
}

/// A population of neural networks undergoing evolutionary training.
/// Designed as immutable to maintain a single memory structure without copying.
pub struct Group {
    individuals: Vec<Individual>,
}

impl Group {
    /// Creates a new population with 256 randomly initialized individuals.
    pub fn new(pop_size: usize) -> Self {
        Self {
            individuals: (0..pop_size)
                .into_iter()
                .map(|_| Individual::default())
                .collect(),
        }
    }

    pub fn individuals(&self) -> &[Individual] {
        &self.individuals
    }

    pub fn individuals_mut(&mut self) -> &mut [Individual] {
        &mut self.individuals
    }

    pub fn inject_weights(&mut self, weights: &[f32], elites: usize, pop_size: usize) {
        for individual in self.individuals.iter_mut() {
            individual.weights.copy_from_slice(weights);
        }
        self.mutate(elites, pop_size);
    }

    /// Mutates all individuals in the population in parallel.
    pub fn mutate(&mut self, elites: usize, pop_size: usize) {
        let (non_elites, elites_slice) = self.individuals.split_at_mut(pop_size - elites);
        non_elites.iter_mut().for_each(|individual| {
            let elite_idx = rand::rng().random_range(..elites);
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
    /// Each individual fights against randomly chosen opponents.
    pub fn train(&mut self, tournament_size: usize) -> usize {
        let len = self.individuals.len();
        // Thread-local Game for each thread
        #[cfg(not(target_arch = "wasm32"))]
        let games = ThreadLocal::new();
        #[cfg(target_arch = "wasm32")]
        let mut game = Game::default();
        // Store (fitness_delta, longest_match_ticks) for each individual
        let run_tournament = |i| {
            let (before, rest) = self.individuals.split_at(i);
            let (current, after) = rest.split_first().unwrap();
            // Select random opponents from remaining individuals
            let others: Vec<&Individual> = before
                .iter()
                .chain(after.iter())
                .choose_multiple(&mut rand::rng(), tournament_size - 1);
            // Use the thread's own Game instance
            let mut delta = 0;
            let mut indiv_longest = 0;
            for (j, &opponent) in others.iter().enumerate() {
                let game: &mut Game = {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        &mut *games.get_or(|| RefCell::new(Game::default())).borrow_mut()
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        &mut game
                    }
                };

                *game = Game::default();
                let (winner, ticks) = game.run_until(
                    current,
                    opponent,
                    if j % 2 == 0 { Side::Left } else { Side::Right },
                );
                indiv_longest = indiv_longest.max(ticks);
                delta += match winner {
                    Side::Left => 1,
                    Side::Right => -1,
                    Side::Neither => 0,
                };
            }

            (delta, indiv_longest)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<(i8, usize)> = (0..len).into_par_iter().map(run_tournament).collect();

        #[cfg(target_arch = "wasm32")]
        let results: Vec<(i8, usize)> = (0..len).into_iter().map(run_tournament).collect();

        // Second pass: apply fitness deltas
        for (ind, (delta, _)) in self.individuals.iter_mut().zip(&results) {
            ind.fitness += *delta;
        }
        let longest_match_ticks = results.iter().map(|(_, ticks)| *ticks).max().unwrap_or(0);
        longest_match_ticks
    }
}
