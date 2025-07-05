//! Application configuration.



// The engine to use for the simulation
#[derive(Debug, Clone, Copy)]
/// Selects which neural network engine/representation to use.
///
/// # Variants
/// - `Stack`: All weights on stack (fast, limited size)
/// - `Simd`: Uses SIMD instructions for parallel math
/// - `ConcurrentStack`: Parallel fitness evaluation (Rayon) with stack-based weights
/// - `ConcurrentSimd`: Parallel fitness evaluation (Rayon) with SIMD-based weights
/// - `Gpu`: GPU-accelerated (wgpu)
/// - `Heap`: Weights on heap (flexible, slower)
/// - `ConcurrentHeap`: Parallel fitness evaluation (Rayon) with heap-based weights
pub enum Engine {
    /// All weights on stack (fast, limited size)
    Stack,
    /// Uses SIMD instructions for parallel math
    Simd,
    /// Parallel fitness evaluation (Rayon) with stack-based weights
    ConcurrentStack,
    /// Parallel fitness evaluation (Rayon) with SIMD-based weights
    ConcurrentSimd,
    /// GPU-accelerated (wgpu)
    Gpu,
    /// Weights on heap (flexible, slower)
    Heap,
    /// Parallel fitness evaluation (Rayon) with heap-based weights
    ConcurrentHeap,
}

impl Engine {
    pub fn to_str(&self) -> &str {
        match self {
            Engine::Stack => "Stack",
            Engine::Simd => "SIMD",
            Engine::ConcurrentStack => "Concurrent Stack",
            Engine::ConcurrentSimd => "Concurrent SIMD",
            Engine::Gpu => "GPU",
            Engine::Heap => "Heap",
            Engine::ConcurrentHeap => "Concurrent Heap",
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Configuration for evolutionary training and neural net engine.
///
/// # Fields
/// - `engine`: Which engine/representation to use.
/// - `activation`: Activation function for all layers.
/// - `mutation_rate`: Probability of mutating a weight.
/// - `mutation_strength`: Magnitude of mutation.
/// - `generations`: Number of generations to evolve.
/// - `concurrent`: Enable parallel fitness evaluation.
/// - `population_size`: Number of individuals in the population.
/// - `elite_count`: Number of elite individuals to select.
pub struct EvolutionConfig {
    /// Engine/representation type
    pub engine: Engine,
    /// Activation function
    pub activation: Activation,
    /// Probability of mutating a weight
    pub mutation_rate: f32,
    /// Magnitude of mutation
    pub mutation_strength: f32,
    /// Number of generations to evolve
    pub generations: u32,
    /// Enable parallel fitness evaluation
    pub concurrent: bool,
    /// Number of individuals in the population
    pub population_size: usize,
    /// Number of elite individuals to select
    pub elite_count: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            generations: 100,
            mutation_rate: 0.05,
            mutation_strength: 0.1,
            engine: Engine::Stack,
            concurrent: false,
            activation: Activation::Tanh,
            population_size: 128,
            elite_count: 11,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Activation function for neural network layers.
///
/// # Variants
/// - `Tanh`: Hyperbolic tangent (smooth, common)
/// - `Relu`: Rectified Linear Unit (fast, sparse)
/// - `Atan`: Arctangent (smooth, less common)
/// - `Linear`: No nonlinearity
pub enum Activation {
    Tanh,
    Relu,
    Atan,
    Linear,
}

impl Activation {
    pub fn to_str(&self) -> &str {
        match self {
            Activation::Tanh => "Tanh",
            Activation::Relu => "ReLU",
            Activation::Atan => "Atan",
            Activation::Linear => "Linear",
        }
    }
}
