use rand::{self, Rng};
use rayon::prelude::*;

const WEIGHTS: usize = 9 * 16 + 17 * 4 + 5 * 1;

struct Individual {
    weights: [f32; WEIGHTS],
}

impl Default for Individual {
    fn default() -> Self {
        let mut array = [0.0; WEIGHTS];
        rand::rng().fill(&mut array);
        Self { weights: array }
    }
}

impl Individual {
    fn mutate(&mut self) {
        for weight in self.weights.iter_mut() {
            *weight = rand::rng().random();
        }
    }
}

pub struct Population {
    individuals: Vec<Individual>,
}

impl Population {
    pub fn new() -> Self {
        Self {
            individuals: (0..256)
                .into_iter()
                .map(|_| Individual::default())
                .collect(),
        }
    }

    pub fn mutate(&mut self) {
        for individual in self.individuals.iter_mut() {
            let mut old = Individual {
                weights: individual.weights.clone(),
            };
            old.mutate();
            *individual = old;
        }
    }
}

pub fn train() {
    let mut pop = Population::new();
    for _ in 0..100 {
        pop.mutate();
    }
}
