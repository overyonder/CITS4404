//! Defines the core `Individual` trait for all neural network implementations.
//!
//! This module provides the essential abstraction that allows the genetic algorithm
//! to work with various neural network 'engines' (like stack-based, heap-based, SIMD)
//! in a uniform way.

use crate::config::{Activation, Config};
use crate::constants::{INPUT_SIZE, OUTPUT_SIZE, TOTAL_WEIGHTS};
use rand::Rng;
use std::io::Write;

/// Defines the required behavior for any neural network implementation (an "individual").
///
/// # Trait Bounds
/// - `Default`: Allows for the creation of a new, default-initialized individual (e.g., with zeroed weights).
/// - `Clone`: Enables creating copies of individuals, essential for breeding new generations.
/// - `Send + Sync`: Marks the type as safe to be sent and shared across threads, a requirement for concurrent evolution.
///
/// # Teaching Note
/// This trait is a prime example of **polymorphism** in Rust. By defining a common interface,
/// we can write the main evolutionary loop once, and it will work with any `Individual` type.
/// This makes the system highly extensible, allowing new network engines to be added without
/// changing the core genetic algorithm logic.
pub trait Individual: Default + Clone + Send + Sync {
    /// Returns a human-readable name for the individual's implementation type.
    /// Used for display purposes in the UI and logs.
    fn name() -> &'static str
    where
        Self: Sized;

    /// Performs the neural network's forward propagation.
    ///
    /// Takes the game state as input and returns the network's output (e.g., paddle movement).
    fn forward_propagate(
        &self,
        input: &[f32; INPUT_SIZE],
        activation: Activation,
    ) -> [f32; OUTPUT_SIZE];

    /// Creates a new child individual by performing crossover between two parents.
    ///
    /// # Teaching Note
    /// Crossover (or recombination) is a fundamental genetic operator. This function implements
    /// uniform crossover: for each weight, it's randomly chosen from one of the two parents.
    /// This allows the child to inherit a mix of traits from both.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let mut child = Self::default();
        let parent1_weights = self.weights_as_slice();
        let parent2_weights = other.weights_as_slice();
        let child_weights = child.weights_as_mut_slice();

        for i in 0..TOTAL_WEIGHTS {
            child_weights[i] = if rng.gen::<bool>() {
                parent1_weights[i]
            } else {
                parent2_weights[i]
            };
        }
        child
    }

    /// Applies mutation to the individual's weights.
    ///
    /// # Teaching Note
    /// Mutation introduces new genetic material into the population, preventing stagnation.
    /// This implementation iterates through each weight and, with a small probability
    /// (`mutation_rate`), perturbs the weight by a random amount (`mutation_strength`).
    fn mutate<R: Rng>(&mut self, rng: &mut R, config: &Config) {
        let weights = self.weights_as_mut_slice();
        for i in 0..TOTAL_WEIGHTS {
            if rng.gen::<f32>() < config.mutation_rate {
                weights[i] += rng.gen_range(-1.0..=1.0) * config.mutation_strength;
            }
        }
    }

    /// Provides a read-only view of the individual's weights (its "genome").
    fn weights_as_slice(&self) -> &[f32];

    /// Provides a mutable view of the individual's weights.
    fn weights_as_mut_slice(&mut self) -> &mut [f32];

    /// Saves the individual's weights to a binary file.
    ///
    /// # Safety
    /// The `unsafe` block is used for a highly efficient serialization. It is safe because:
    /// 1. We are converting a slice of `f32` into a slice of bytes (`u8`).
    /// 2. `f32` has a stable, platform-independent memory representation (IEEE 754).
    /// 3. The lifetime of the byte slice is tied to the weight slice, ensuring no dangling pointers.
    /// This avoids slower, byte-by-byte serialization.
    fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let weights_slice = self.weights_as_slice();
        assert_eq!(
            weights_slice.len(),
            TOTAL_WEIGHTS,
            "Weight slice length mismatch."
        );
        let weights_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights_slice.as_ptr() as *const u8,
                weights_slice.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(weights_bytes)
    }
}
