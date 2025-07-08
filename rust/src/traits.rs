//! Defines the core `Individual` trait for all neural network implementations.
//!
//! This module provides the essential abstraction that allows the genetic algorithm
//! to work with various neural network 'engines' (like stack-based, heap-based, SIMD)
//! in a uniform way. This is a key example of the **Strategy Pattern** in action.

use crate::config::{Activation, Config};
use crate::constants::{INPUT_SIZE, OUTPUT_SIZE, TOTAL_WEIGHTS};
use rand::Rng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

/// Serializable representation of a trained individual for storage.
///
/// # Teaching Note
/// This struct separates the data representation from the behavior, following the
/// **Single Responsibility Principle**. The Individual trait handles neural network
/// operations, while this struct handles serialization concerns.
#[derive(Serialize, Deserialize)]
pub struct SerializableIndividual {
    /// The neural network weights (the "genome" in genetic algorithm terms)
    pub weights: Vec<f32>,
    /// The configuration used during training (hyperparameters)
    pub config: Config,
}

/// Defines the required behavior for any neural network implementation (an "individual").
///
/// # Trait Bounds
/// - `Default`: Allows for the creation of a new, default-initialized individual (e.g., with zeroed weights).
/// - `Clone`: Enables creating copies of individuals, essential for breeding new generations.
/// - `Send + Sync`: Marks the type as safe to be sent and shared across threads, a requirement for concurrent evolution.
///
/// # Teaching Note: Polymorphism and Abstraction
/// This trait is a prime example of **polymorphism** in Rust. By defining a common interface,
/// we can write the main evolutionary loop once, and it will work with any `Individual` type.
/// This makes the system highly extensible, allowing new network engines to be added without
/// changing the core genetic algorithm logic. This pattern is especially important in
/// machine learning where different compute backends (CPU, GPU, SIMD) may be optimal
/// for different scenarios.
pub trait Individual: Default + Clone + Send + Sync {
    /// Performs the neural network's forward propagation.
    ///
    /// # Neural Network Forward Pass Algorithm:
    /// 1. Take input layer activations
    /// 2. For each hidden layer:
    ///    a. Compute weighted sum: output = Σ(input_i × weight_i) + bias
    ///    b. Apply activation function: output = activation(weighted_sum)
    /// 3. Return final layer outputs
    ///
    /// # Teaching Note: Universal Approximation
    /// Neural networks with at least one hidden layer can approximate any continuous function
    /// to arbitrary precision (Universal Approximation Theorem). The choice of activation
    /// function affects convergence speed and the types of functions that can be learned efficiently.
    fn forward_propagate(
        &self,
        input: &[f32; INPUT_SIZE],
        activation: Activation,
    ) -> [f32; OUTPUT_SIZE];

    /// Creates a new child individual by performing crossover between two parents.
    ///
    /// # Genetic Algorithm: Crossover (Recombination) Operator
    /// Crossover combines genetic material from two parents to create offspring.
    /// Common strategies include:
    /// - **Uniform Crossover** (implemented here): Each gene randomly chosen from either parent
    /// - **Single-Point Crossover**: Split at one point, take prefix from parent1, suffix from parent2
    /// - **Two-Point Crossover**: Split at two points, alternate between parents
    /// - **Arithmetic Crossover**: Weighted average of parent values
    ///
    /// # Teaching Note: Exploration vs Exploitation
    /// Crossover balances **exploration** (trying new combinations) with **exploitation**
    /// (keeping good traits). Uniform crossover provides fine-grained mixing, which can
    /// be especially effective for neural networks where individual weights may be
    /// independently important.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let mut child = self.clone();
        let parent2_weights = other.weights_as_slice();
        let child_weights = child.weights_as_mut_slice();

        for i in 0..TOTAL_WEIGHTS {
            // 50% chance to take a given weight from the second parent
            if rng.random::<bool>() {
                child_weights[i] = parent2_weights[i];
            }
        }
        child
    }

    /// Applies mutation to the individual's weights.
    ///
    /// # Genetic Algorithm: Mutation Operator
    /// Mutation introduces new genetic material into the population, serving several purposes:
    /// 1. **Prevents Premature Convergence**: Maintains genetic diversity
    /// 2. **Escapes Local Optima**: Enables exploration of new solution regions
    /// 3. **Hill Climbing**: Makes small improvements to existing solutions
    ///
    /// # Mutation Strategies:
    /// - **C++ Equivalent**: Single gene mutation with Gaussian noise (more conservative)
    /// - **Modern**: Multiple gene mutation with uniform rate (more exploratory)
    ///
    /// # Teaching Note: Mutation Rate Tuning
    /// - Too high: Population becomes random, loses good solutions
    /// - Too low: Population stagnates, slow convergence
    /// - Typical range: 0.01-0.1 (1%-10% of genes mutated)
    fn mutate<R: Rng>(&mut self, rng: &mut R, config: &Config) {
        let weights = self.weights_as_mut_slice();
        
        match config.mutation_strategy {
            crate::config::MutationStrategy::CppEquivalent => {
                // Conservative strategy: mutate exactly one randomly selected gene
                // Uses Gaussian distribution N(0, 1) for biologically-inspired variation
                let gene_index = rng.random_range(0..TOTAL_WEIGHTS);
                let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
                let mutation = normal.sample(rng);
                weights[gene_index] += mutation;
            }
            crate::config::MutationStrategy::Modern => {
                // Modern strategy: probabilistic mutation of multiple genes
                // Each gene has `mutation_rate` chance of being perturbed by `mutation_strength`
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
    /// # Teaching Note: Abstraction and Encapsulation
    /// This method is a crucial piece of the abstraction. It allows the default `crossover`
    /// and `mutate` methods to work on the individual's weights without needing to know
    /// *how* those weights are stored (e.g., in a stack array for `StackIndividual` or a
    /// heap-allocated `Vec` for `HeapIndividual`). This separation of interface from
    /// implementation is a core principle of object-oriented design.
    fn weights_as_slice(&self) -> &[f32];

    /// Provides a mutable view of the individual's weights.
    fn weights_as_mut_slice(&mut self) -> &mut [f32];

    /// Saves the individual's weights and configuration to a JSON file.
    ///
    /// # JSON Serialization Benefits:
    /// - **Human Readable**: Easy to inspect and debug trained models
    /// - **Language Agnostic**: Can be loaded by any language with JSON support
    /// - **Version Safe**: Field names make format more robust to changes
    /// - **Debuggable**: Can manually edit models for experimentation
    ///
    /// # File Format:
    /// ```json
    /// {
    ///   "weights": [0.1, -0.5, 0.3, ...],
    ///   "config": {
    ///     "population_size": 128,
    ///     "mutation_rate": 0.05,
    ///     ...
    ///   }
    /// }
    /// ```
    ///
    /// # Teaching Note: Trade-offs
    /// JSON is larger than binary but provides significant advantages for educational
    /// and research contexts where model interpretability matters more than storage efficiency.
    fn save(&self, path: &str, config: &Config) -> std::io::Result<()> {
        // Create a mutable copy of the config and set the timestamp
        let mut config_to_save = config.clone();
        config_to_save.date_trained = Some(chrono::Utc::now());

        let serializable = SerializableIndividual {
            weights: self.weights_as_slice().to_vec(),
            config: config_to_save,
        };

        let json = serde_json::to_string_pretty(&serializable)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())
    }
}
