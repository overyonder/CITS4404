use rand::{self, Rng};
use std::time::Instant;

const WEIGHTS: usize = 9 * 16 + 17 * 4 + 5 * 1;

struct Individual {
    weights: [f32; WEIGHTS],
}

impl Individual {
    fn new_fill() -> Self {
        let mut array = [0.0; WEIGHTS];
        rand::rng().fill(&mut array);
        Self { weights: array }
    }

    fn new_from_fn() -> Self {
        let mut rng = rand::rng();
        let weights = std::array::from_fn(|_| rng.random());
        Self { weights: weights }
    }
}

struct Population {
    individuals: [Individual; 100],
}

impl Population {
    fn new_fill() -> Self {
        Self {
            individuals: std::array::from_fn(|_| Individual::new_fill()),
        }
    }
    fn new_from_fn() -> Self {
        Self {
            individuals: std::array::from_fn(|_| Individual::new_from_fn()),
        }
    }
}

fn main() {
    let earlier = Instant::now();
    benchmark_fill();
    println!("{:?}", Instant::now().duration_since(earlier));
}

fn benchmark_fill() {
    for _ in 0..10000 {
        let _pop = Population::new_fill();
    }
}

fn benchmark_from() {
    // Set up benchmarking?
    for _ in 0..10000 {
        let _pop = Population::new_from_fn();
    }
    // Report?
}
