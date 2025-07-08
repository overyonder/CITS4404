//! Defines the core configuration structures for the application.
//!
//! This module contains the enums and structs that control the behavior of the
//! evolutionary algorithm and the neural network engines. It is designed to be
//! clear, well-documented, and easy to extend.
//!
//! # Teaching Note: Configuration Design Patterns
//! This module demonstrates several important software engineering principles:
//! - **Single Source of Truth**: All parameters are centralized in one place
//! - **Type Safety**: Enums prevent invalid configuration values
//! - **Extensibility**: New parameters and options can be added without breaking existing code
//! - **Documentation as Code**: Each parameter includes its purpose and impact

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Selects the underlying engine for neural network representation and computation.
///
/// # Teaching Note: Engine Selection Trade-offs
/// Each engine represents a different approach to the classic **time vs space vs parallelism** trade-off:
/// - **CPU**: Optimized for single-threaded performance and memory efficiency
/// - **GPU**: Optimized for massive parallelism at the cost of memory bandwidth
///
/// The choice affects not just performance but also debugging capabilities, as GPU
/// kernels are harder to debug than CPU code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Engine {
    /// **CPU Engine:** Uses fixed-size arrays with optimized CPU computation.
    /// - **Advantages**: Excellent cache locality, zero allocation overhead, deterministic performance
    /// - **Best for**: Small to medium populations (< 500), debugging, predictable performance
    /// - **Implementation**: Network size fixed at compile time for maximum optimization
    #[default]
    Cpu,
    /// **GPU Engine:** Offloads the entire tournament evaluation to GPU using WGSL shaders.
    /// - **Advantages**: Massive parallelism (thousands of individuals simultaneously)
    /// - **Best for**: Large populations (500+), when raw throughput matters most
    /// - **Implementation**: Batched evaluation with GPU memory management
    Gpu,
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Engine::Cpu => write!(f, "CPU"),
            Engine::Gpu => write!(f, "GPU"),
        }
    }
}

/// Parent selection strategy for the evolutionary algorithm.
///
/// # Teaching Note: Selection Pressure
/// Selection pressure determines how aggressively the algorithm favors fitter individuals.
/// Too high: rapid convergence but risk of premature convergence to local optima
/// Too low: slow convergence but better exploration of the search space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SelectionStrategy {
    /// **Fitness Proportionate**: Probability of selection proportional to fitness
    /// - Classic "roulette wheel" selection
    /// - Can suffer from premature convergence if one individual dominates
    FitnessProportionate,
    /// **Tournament Selection**: Random groups compete, winner selected
    /// - Most widely used in practice due to good balance of pressure and diversity
    /// - Tournament size controls selection pressure
    #[default]
    Tournament,
    /// **Rank Based**: Selection based on fitness rank, not absolute fitness values
    /// - Maintains consistent selection pressure regardless of fitness distribution
    /// - Prevents single super-fit individual from dominating
    RankBased,
    /// **Truncation**: Only top percentage of population can reproduce
    /// - Highest selection pressure, fastest convergence
    /// - Risk of losing diversity quickly
    Truncation,
}

impl fmt::Display for SelectionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectionStrategy::FitnessProportionate => write!(f, "Fitness Proportionate"),
            SelectionStrategy::Tournament => write!(f, "Tournament"),
            SelectionStrategy::RankBased => write!(f, "Rank Based"),
            SelectionStrategy::Truncation => write!(f, "Truncation"),
        }
    }
}

/// Selects the fitness function for the evolutionary algorithm.
///
/// # Teaching Note: Fitness Function Design
/// The fitness function is arguably the most important component of any evolutionary algorithm.
/// It defines the optimization objective and directly influences the behavior that emerges.
/// Good fitness functions are:
/// - **Smooth**: Small changes in genome produce small changes in fitness
/// - **Multi-modal**: Reward multiple strategies to maintain diversity
/// - **Scalable**: Distinguish between good and great solutions
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize, Default)]
pub enum FitnessFunc {
    /// **C++ Equivalent:** Rewards survival time (frames) and successful returns
    /// - Formula: survival_time + successful_returns * bonus
    /// - **Pros**: Direct port for performance comparison, rewards defensive play
    /// - **Cons**: May reward stalling rather than decisive play
    #[default]
    CppEquivalent,
    /// **Return Focused:** Primarily rewards successful returns with small winning bonus
    /// - Formula: successful_returns * 10 + wins * 5 + survival_time * 0.1
    /// - **Pros**: Encourages active, engaging gameplay
    /// - **Cons**: May undervalue defensive strategies
    ReturnFocused,
    /// **Victory Optimized:** Strongly rewards decisive wins and rally duration
    /// - Formula: wins * 50 + rally_length * 2 + successful_returns * 5
    /// - **Pros**: Encourages aggressive, dominating play
    /// - **Cons**: May neglect consistent but less spectacular strategies
    VictoryOptimized,
}

impl fmt::Display for FitnessFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitnessFunc::CppEquivalent => write!(f, "C++ Equivalent"),
            FitnessFunc::ReturnFocused => write!(f, "Return Focused"),
            FitnessFunc::VictoryOptimized => write!(f, "Victory Optimized"),
        }
    }
}

/// Holds all parameters for an evolutionary training session.
///
/// # Teaching Note: Configuration as Code
/// This struct demonstrates the **Configuration as Code** pattern, where all system
/// behavior is controlled through explicit, serializable parameters. This approach:
/// - Makes experiments reproducible
/// - Enables systematic parameter sweeps
/// - Facilitates sharing successful configurations
/// - Provides audit trails for research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // === MODEL METADATA ===
    /// A user-friendly name for the model, often the filename
    #[serde(default)]
    pub name: Option<String>,
    /// The UTC date and time when the model was saved
    #[serde(default)]
    pub date_trained: Option<DateTime<Utc>>,

    // === CORE ALGORITHM PARAMETERS ===
    /// The neural network engine to use for forward propagation
    pub engine: Engine,
    /// The activation function for hidden and output layers
    pub activation: Activation,
    /// The total number of generations to run the evolution for
    /// Each generation = evaluation + selection + reproduction + mutation
    pub generations: u32,
    /// If true, fitness evaluation parallelized across available CPU cores using Rayon
    /// Provides significant speedup for CPU engines, no effect on GPU engine
    pub concurrent: bool,

    // === POPULATION PARAMETERS ===
    /// The number of individuals in the population
    /// **Trade-off**: Larger = more diversity but higher computational cost per generation
    /// **Typical range**: 50-1000, sweet spot often 100-300
    pub population_size: usize,
    /// The number of highest-scoring individuals to carry over unchanged (elitism)
    /// **Purpose**: Ensures best solutions are never lost due to random variation
    /// **Typical range**: 1-10% of population size
    pub elite_count: usize,

    // === SELECTION PARAMETERS ===
    /// The strategy for selecting parents for reproduction
    pub selection_strategy: SelectionStrategy,
    /// Tournament size for tournament selection (ignored for other strategies)
    /// **Effect**: Larger = higher selection pressure, faster convergence
    /// **Typical range**: 2-7, most common is 3-5
    pub tournament_size: usize,
    /// Percentage of population that can reproduce (for truncation selection)
    /// **Effect**: Lower = higher pressure, only best parents reproduce
    /// **Typical range**: 0.1-0.5 (10%-50%)
    pub truncation_rate: f32,

    // === REPRODUCTION PARAMETERS ===
    /// The strategy for creating a new generation from parents
    pub reproduction_strategy: ReproductionStrategy,
    /// Probability that crossover occurs vs copying parent (0.0-1.0)
    /// **Effect**: Higher = more genetic mixing, potentially faster convergence
    /// **Typical range**: 0.6-0.95, most common 0.8-0.9
    pub crossover_rate: f32,

    // === MUTATION PARAMETERS ===
    /// The strategy for mutating individuals during evolution
    pub mutation_strategy: MutationStrategy,
    /// The probability (0.0-1.0) of a single gene (weight) being mutated
    /// **Effect**: Higher = more exploration but can destabilize good solutions
    /// **Typical range**: 0.01-0.1 (1%-10%)
    pub mutation_rate: f32,
    /// The maximum value to add/subtract during mutation
    /// **Effect**: Higher = larger search steps, can escape local optima faster
    /// **Typical range**: 0.01-0.5, often 10% of typical weight magnitude
    pub mutation_strength: f32,
    /// If true, mutation rate adapts based on population diversity
    /// **Purpose**: High mutation when population converges, low when diverse
    pub adaptive_mutation: bool,
    /// Minimum mutation rate for adaptive mutation
    pub min_mutation_rate: f32,
    /// Maximum mutation rate for adaptive mutation  
    pub max_mutation_rate: f32,

    // === CONVERGENCE AND STOPPING CRITERIA ===
    /// Stop if best fitness doesn't improve for this many generations
    /// **Purpose**: Saves computation when algorithm has plateaued
    /// **Typical range**: 10-50 generations
    pub early_stopping_patience: Option<u32>,
    /// Stop if best fitness improvement is below this threshold
    /// **Purpose**: Defines "meaningful" improvement to avoid infinite tiny gains
    pub fitness_threshold: Option<f32>,
    /// If true, tracks population diversity metrics during evolution
    /// **Purpose**: Helps diagnose premature convergence issues
    pub track_diversity: bool,

    // === FITNESS FUNCTION PARAMETERS ===
    /// The fitness function to use for evaluation
    pub fitness_func: FitnessFunc,
    /// If true, fitness values are normalized to [0,1] range each generation
    /// **Purpose**: Maintains consistent selection pressure across generations
    pub normalize_fitness: bool,

    // === SIMULATION PARAMETERS ===
    /// If true, ball starting velocity randomized rather than fixed
    /// **Purpose**: Adds variability to training scenarios, prevents overfitting
    pub random_ball_direction: bool,
    /// Random seed for reproducible experiments (None = random seed)
    /// **Purpose**: Enables exact reproduction of training runs for research
    pub random_seed: Option<u64>,
    /// Simulation speed multiplier for adjustable training visualization
    /// **Range**: 0.1-10.0, where 1.0 = normal speed, 2.0 = 2x faster, 0.5 = half speed
    /// **Purpose**: Allows fine-tuning of training visualization speed for optimal observation
    pub simulation_speed: f32,
}

/// Provides a scientifically sound default configuration for the evolutionary algorithm.
///
/// # Teaching Note: Default Parameter Selection
/// These defaults represent a balance between:
/// - **Exploration vs Exploitation**: Parameters that maintain diversity while making progress
/// - **Computational Efficiency**: Settings that work well across different hardware
/// - **Research Best Practices**: Values commonly used in EA literature
/// - **C++ Compatibility**: Matching the original implementation where beneficial
impl Default for Config {
    fn default() -> Self {
        Self {
            // Model metadata
            name: None,
            date_trained: None,

            // Core algorithm
            engine: Engine::default(),
            activation: Activation::ClampedLinear, // C++ equivalent
            generations: 5000,
            concurrent: false,

            // Population parameters
            population_size: 128,     // C++ default
            elite_count: 11,          // C++ default: ~8.6% of population

            // Selection parameters
            selection_strategy: SelectionStrategy::Tournament,
            tournament_size: 3,       // Moderate selection pressure
            truncation_rate: 0.3,     // Top 30% can reproduce

            // Reproduction parameters
            reproduction_strategy: ReproductionStrategy::default(),
            crossover_rate: 0.8,      // Standard value in EA literature

            // Mutation parameters
            mutation_strategy: MutationStrategy::default(),
            mutation_rate: 0.05,      // C++ default: 5%
            mutation_strength: 0.1,   // C++ default: ±10%
            adaptive_mutation: false,
            min_mutation_rate: 0.01,  // 1% minimum
            max_mutation_rate: 0.2,   // 20% maximum

            // Convergence criteria
            early_stopping_patience: None, // No early stopping by default
            fitness_threshold: Some(0.01),     // 1% improvement threshold (updated range)
            track_diversity: false,

            // Fitness function
            fitness_func: FitnessFunc::CppEquivalent,
            normalize_fitness: false,

            // Simulation
            random_ball_direction: false, // C++ default: fixed direction
            random_seed: None,           // Random seed each run
            simulation_speed: 1.0,       // Normal speed (1x multiplier)
        }
    }
}

/// The non-linear activation function used in the neural network's layers.
///
/// # Teaching Note: Activation Function Theory
/// Activation functions serve several critical purposes in neural networks:
/// 1. **Non-linearity**: Without them, the network is just linear regression
/// 2. **Gradient Flow**: Affects how well gradients propagate during learning
/// 3. **Output Range**: Controls the range of neuron outputs
/// 4. **Computational Cost**: Some functions are much faster to compute
///
/// For evolutionary algorithms (no backprop), gradient properties matter less,
/// so we can focus on expressiveness and computational efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Activation {
    /// **Clamped Linear:** `f(x) = clamp(x, -1, 1)`
    /// - **Range**: [-1, 1]
    /// - **Properties**: Simple, fast, maintains input relationships
    /// - **Use case**: C++ compatibility, simple problems
    #[default]
    ClampedLinear,
    /// **Hyperbolic Tangent:** `f(x) = tanh(x)`
    /// - **Range**: (-1, 1)
    /// - **Properties**: Smooth, zero-centered, well-behaved gradients
    /// - **Use case**: Most neural network applications, good default choice
    Tanh,
    /// **Rectified Linear Unit:** `f(x) = max(0, x)`
    /// - **Range**: [0, ∞)
    /// - **Properties**: Very fast, helps with vanishing gradients, not zero-centered
    /// - **Use case**: Deep networks, when sparsity is desired
    Relu,
    /// **Arctangent:** `f(x) = atan(x)`
    /// - **Range**: (-π/2, π/2)
    /// - **Properties**: Similar to tanh but different output range
    /// - **Use case**: Alternative to tanh when different scaling needed
    Atan,
    /// **Sigmoid:** `f(x) = 1 / (1 + e^(-x))`
    /// - **Range**: (0, 1)
    /// - **Properties**: Smooth, interpretable as probability, can suffer from saturation
    /// - **Use case**: Binary classification, when outputs should be probabilities
    Sigmoid,
    /// **Linear:** `f(x) = x`
    /// - **Range**: (-∞, ∞)
    /// - **Properties**: No non-linearity, turns network into linear regression
    /// - **Use case**: Output layers, debugging, baseline comparison
    Linear,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::ClampedLinear => write!(f, "Clamped Linear"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::Relu => write!(f, "ReLU"),
            Activation::Atan => write!(f, "Atan"),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Linear => write!(f, "Linear"),
        }
    }
}

/// Selects the algorithm for generating a new population from the fittest members.
///
/// # Teaching Note: Reproduction Strategies
/// The reproduction strategy determines how genetic material flows from one generation
/// to the next. This is a key component of the **genetic algorithm lifecycle**:
/// 1. **Evaluation**: Measure fitness of all individuals
/// 2. **Selection**: Choose which individuals become parents
/// 3. **Reproduction**: Create offspring using this strategy
/// 4. **Mutation**: Add random variation to offspring
/// 5. **Replacement**: Form new population
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReproductionStrategy {
    /// **C++ Equivalent:** Top √N individuals survive and reproduce
    /// - **Algorithm**: 
    ///   1. Select top √N fittest individuals as survivors
    ///   2. Cross every pair of survivors to create offspring
    ///   3. Fill remaining slots with mutated copies of survivors
    /// - **Properties**: Moderate selection pressure, deterministic survivor count
    #[default]
    CppEquivalent,
    /// **Elite Crossover:** Configurable elitism with random parent selection
    /// - **Algorithm**:
    ///   1. Carry over `elite_count` best individuals unchanged
    ///   2. For remaining slots: randomly select 2 parents from elites
    ///   3. Create offspring via crossover and mutation
    /// - **Properties**: Higher genetic diversity, configurable selection pressure
    EliteCrossover,
}

impl fmt::Display for ReproductionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReproductionStrategy::CppEquivalent => write!(f, "C++ Equivalent"),
            ReproductionStrategy::EliteCrossover => write!(f, "Elite Crossover"),
        }
    }
}

/// Selects the mutation strategy for the evolutionary algorithm.
///
/// # Teaching Note: Mutation in Evolutionary Algorithms
/// Mutation serves as the **exploration mechanism** in evolutionary algorithms.
/// Unlike crossover (which recombines existing genetic material), mutation introduces
/// entirely new genetic material into the population. The balance between mutation
/// and crossover determines the exploration-exploitation trade-off.
///
/// ## Mutation Rate Guidelines:
/// - **Too High (>0.1)**: Population becomes random walk, loses good solutions
/// - **Too Low (<0.001)**: Population stagnates, slow to escape local optima  
/// - **Just Right (0.01-0.1)**: Maintains diversity while preserving good solutions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
pub enum MutationStrategy {
    /// **C++ Equivalent:** Single gene Gaussian mutation
    /// - **Algorithm**: Select one random gene, add N(0,1) noise
    /// - **Properties**: Conservative, biologically inspired, deterministic mutation count
    /// - **Best for**: Maintaining population stability, fine-tuning solutions
    #[default]
    CppEquivalent,
    /// **Modern:** Probabilistic multi-gene mutation
    /// - **Algorithm**: Each gene has `mutation_rate` chance of uniform perturbation
    /// - **Properties**: More exploratory, configurable intensity
    /// - **Best for**: Escaping local optima, early exploration phases
    Modern,
}

impl fmt::Display for MutationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MutationStrategy::CppEquivalent => write!(f, "C++ Equivalent"),
            MutationStrategy::Modern => write!(f, "Modern"),
        }
    }
}
