//! Defines the core configuration structures for the application.
//!
//! This module contains the enums and structs that control the behavior of the
//! evolutionary algorithm and the neural network engines. It is designed to be
//! clear, well-documented, and easy to extend.

use std::fmt;

/// Selects the underlying engine for neural network representation and computation.
///
/// Each engine has different performance characteristics and memory allocation strategies.
/// The choice of engine is a trade-off between raw speed, memory usage, and flexibility.
///
/// # Teaching Note
/// This enum demonstrates how to represent a set of mutually exclusive choices. The `Display`
/// trait is implemented for user-friendly string representation, which is more idiomatic
/// than a custom `to_str()` method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// **Stack-allocated:** Uses fixed-size arrays on the stack for genomes. This is extremely
    /// fast due to superior cache locality and zero allocation overhead, but the network size
    /// is fixed at compile time.
    Stack,
    /// **Heap-allocated:** Uses `Vec<f32>` on the heap for genomes. This is more flexible,
    /// allowing for dynamic network sizes, but may be slower due to heap allocation
    /// overhead and potential cache misses.
    Heap,
    /// **SIMD-optimized:** A stack-based engine that uses Single Instruction, Multiple Data
    /// (SIMD) intrinsics to perform parallel computations on weight data. This can provide
    /// a significant speed boost on compatible CPUs.
    Simd,
    /// **GPU-accelerated:** Offloads the entire tournament evaluation to the GPU using
    /// WGSL shaders. This is the most powerful option for large populations, leveraging
    /// massive parallelism.
    Gpu,
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Engine::Stack => write!(f, "Stack"),
            Engine::Heap => write!(f, "Heap"),
            Engine::Simd => write!(f, "SIMD"),
            Engine::Gpu => write!(f, "GPU"),
        }
    }
}

/// Holds all parameters for an evolutionary training session.
///
/// This struct consolidates all settings, making it easy to pass configuration
/// throughout the application. It derives `Copy` because all its fields are simple
/// types, making it cheap to duplicate.
///
/// # Teaching Note
/// This is a classic example of the **Builder Pattern**'s cousin. By implementing the
/// `Default` trait, we provide a sensible baseline configuration. Users can then
/// instantiate a default and modify only the fields they care about, either in code
/// or through the CLI parser.
#[derive(Debug, Clone, Copy)]
pub struct EvolutionConfig {
    /// The neural network engine to use.
    pub engine: Engine,
    /// The activation function for hidden and output layers.
    pub activation: Activation,
    /// The probability (0.0 to 1.0) of a single gene (weight) being mutated.
    /// A higher value increases genetic diversity and exploration but can destabilize
    /// well-performing genomes.
    pub mutation_rate: f32,
    /// The maximum value to add or subtract during mutation. A higher strength allows
    /// for larger leaps in the search space, potentially escaping local optima faster,
    /// but can also overshoot good solutions.
    pub mutation_strength: f32,
    /// The total number of generations to run the evolution for. Each generation is one
    /// cycle of evaluation, selection, and reproduction.
    pub generations: u32,
    /// If `true`, fitness evaluation is parallelized across available CPU cores using Rayon.
    /// This provides a significant speedup for CPU-bound engines (`Stack`, `Heap`, `Simd`).
    pub concurrent: bool,
    /// The number of individuals in the population. A larger population maintains more
    /// genetic diversity, reducing the risk of premature convergence, but increases the
    /// computational cost per generation.
    pub population_size: usize,
    /// The number of the highest-scoring individuals to carry over to the next generation
    /// without modification. Elitism ensures that the best-found solutions are never lost.
    pub elite_count: usize,
}

/// Provides a default, sensible configuration for the evolutionary algorithm.
///
/// # Teaching Note
/// The `Default` trait is a standard Rust interface for providing a default value for a type.
/// It's widely used in the ecosystem and makes types easier to work with, especially in
/// configuration contexts like this one.
impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            generations: 100,
            mutation_rate: 0.05,      // 5% chance to mutate a gene
            mutation_strength: 0.1, // Mutate by up to +/- 10%
            engine: Engine::Stack,
            concurrent: false,
            activation: Activation::Tanh,
            population_size: 128,
            elite_count: 2, // Keep the top 2 individuals
        }
    }
}

/// The non-linear activation function used in the neural network's layers.
///
/// # Teaching Note
/// Enums are perfect for representing a fixed set of choices. Implementing `Display`
/// makes them easy to print in the UI and logs. `PartialEq` and `Eq` allow them to be
/// compared, which is useful in the TUI for showing the selected item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// **Hyperbolic Tangent:** A smooth, zero-centered function that squashes values to `[-1, 1]`.
    /// Good for general purpose use.
    Tanh,
    /// **Rectified Linear Unit:** Outputs `max(0, x)`. It's computationally very efficient
    /// and helps with the vanishing gradient problem, but it's not zero-centered.
    Relu,
    /// **Arctangent:** Similar to Tanh, it's a smooth function that squashes values,
    /// but to the range `[-PI/2, PI/2]`.
    Atan,
    /// **Linear:** A no-op function (`f(x) = x`). Using this removes non-linearity, turning
    /// the neural network into a simple linear regression model.
    Linear,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::Tanh => write!(f, "Tanh"),
            Activation::Relu => write!(f, "ReLU"),
            Activation::Atan => write!(f, "Atan"),
            Activation::Linear => write!(f, "Linear"),
        }
    }
}
