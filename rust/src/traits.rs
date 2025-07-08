//! Defines the core `Individual` trait for all neural network implementations.
//!
//! This module provides the essential abstraction that allows the genetic algorithm
//! to work with various neural network 'engines' (like stack-based, heap-based, SIMD)
//! in a uniform way.

use crate::config::{Activation, Config};
use crate::constants::{INPUT_SIZE, OUTPUT_SIZE, TOTAL_WEIGHTS};
use rand::Rng;
use std::fs::File;
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
    /// Crossover (or recombination) is a fundamental genetic operator. This default
    /// implementation uses **uniform crossover**: for each weight in the genome, it's
    /// randomly chosen from one of the two parents. This allows the child to inherit a
    /// fine-grained mix of traits. Other strategies, like single-point or two-point
    /// crossover, are also common but uniform crossover is simple and often effective.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let mut child = Self::default();
        let parent1_weights = self.weights_as_slice();
        let parent2_weights = other.weights_as_slice();
        let child_weights = child.weights_as_mut_slice();

        for i in 0..TOTAL_WEIGHTS {
            child_weights[i] = if rng.random::<bool>() {
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
    /// Mutation introduces new genetic material into the population, preventing premature
    /// convergence and exploring new areas of the solution space. This function supports
    /// two strategies:
    /// - **C++ Equivalent**: Mutates exactly one randomly selected gene with normal distribution N(0, 1)
    /// - **Modern**: Mutates multiple genes based on rate and strength parameters
    fn mutate<R: Rng>(&mut self, rng: &mut R, config: &Config) {
        let weights = self.weights_as_mut_slice();
        
        match config.mutation_strategy {
            crate::config::MutationStrategy::CppEquivalent => {
                // C++ equivalent: mutate exactly one randomly selected gene with N(0, 1)
                let gene_index = rng.random_range(0..TOTAL_WEIGHTS);
                let mutation = rng.random_range(-1.0..=1.0); // Normal distribution approximation
                weights[gene_index] += mutation;
            }
            crate::config::MutationStrategy::Modern => {
                // Modern: mutate multiple genes based on rate and strength
                for i in 0..TOTAL_WEIGHTS {
                    if rng.random::<f32>() < config.mutation_rate {
                        weights[i] += rng.random_range(-1.0..=1.0) * config.mutation_strength;
                    }
                }
            }
        }
    }

    /// Provides a read-only view of the individual's weights (its "genome").
    ///
    /// # Teaching Note
    /// This method is a crucial piece of the abstraction. It allows the default `crossover`
    /// and `mutate` methods to work on the individual's weights without needing to know
    /// *how* those weights are stored (e.g., in a stack array for `StackIndividual` or a
    /// heap-allocated `Vec` for `HeapIndividual`).
    fn weights_as_slice(&self) -> &[f32];

    /// Provides a mutable view of the individual's weights.
    fn weights_as_mut_slice(&mut self) -> &mut [f32];

    /// Loads an individual and its configuration from a binary file.
    ///
    /// # Returns
    /// A `Result` containing a tuple of the loaded `Individual` and its `Config`,
    /// or an error if loading fails.
    ///
    /// # Teaching Note
    /// This associated function (like a static method) is responsible for deserialization.
    /// It reads the metadata header first, then the raw weight data, and constructs a new
    /// individual. This is an example of the **Factory Pattern**. Each engine must provide
    /// its own implementation because the way weights are stored internally (e.g., a stack
    /// array vs. a heap vector) differs. The function returns a `Box<dyn std::error::Error>`
    /// to gracefully handle different potential failure modes, like file-not-found I/O
    // errors or malformed JSON parsing errors.
    fn load(path: &str) -> Result<(Self, Config), Box<dyn std::error::Error>>
    where
        Self: Sized;

    /// Saves the individual's weights and configuration to a binary file.
    ///
    /// # File Format
    /// The method serializes the individual into a custom binary format:
    /// 1.  **Config Length (8 bytes)**: A `u64` (little-endian) indicating the size of the
    ///     following JSON configuration string.
    /// 2.  **Config Data (variable size)**: The UTF-8 encoded JSON string of the `Config` struct.
    ///     This stores all hyperparameters used during training.
    /// 3.  **Weight Data (variable size)**: The raw `f32` weights of the neural network, written
    ///     directly from memory.
    ///
    /// # Teaching Note: `unsafe` for Performance
    /// The `unsafe` block is used for a highly efficient serialization of the `f32` weights.
    /// Instead of iterating and writing each float one by one, it reinterprets the `&[f32]`
    /// slice as a `&[u8]` byte slice and writes the entire block of memory at once.
    /// This is considered safe because primitive types like `f32` have a standardized,
    /// stable memory representation (IEEE 754) and no padding bytes, so the direct memory
    /// copy is valid.
    fn save(&self, path: &str, config: &Config) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Create a mutable copy of the config and set the timestamp
        let mut config_to_save = config.clone();
        config_to_save.date_trained = Some(chrono::Utc::now());

        // 1. Serialize config to JSON and write its length and data
        let config_json = serde_json::to_string_pretty(&config_to_save)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let config_bytes = config_json.as_bytes();
        file.write_all(&(config_bytes.len() as u64).to_le_bytes())?;
        file.write_all(config_bytes)?;

        // 2. Write the raw weights
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
