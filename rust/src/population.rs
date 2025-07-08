//! Implements the core genetic algorithm for evolving neural network populations.
//!
//! This module contains the `Population` struct and all associated methods for
//! running evolutionary training. It demonstrates key genetic algorithm concepts
//! including selection, crossover, mutation, and fitness evaluation.
//!
//! # Teaching Note: Genetic Algorithm Architecture
//! This implementation follows the classic genetic algorithm pattern:
//! 1. **Initialization**: Create random population
//! 2. **Evaluation**: Measure fitness of all individuals
//! 3. **Selection**: Choose parents based on fitness
//! 4. **Reproduction**: Create offspring via crossover and mutation
//! 5. **Replacement**: Form new generation
//! 6. **Repeat**: Continue until termination criterion met
//!
//! # Key Concepts Demonstrated:
//! - **Population-based search**: Multiple candidate solutions evolve simultaneously
//! - **Selection pressure**: Fitter individuals have higher reproduction probability  
//! - **Genetic operators**: Crossover combines solutions, mutation introduces novelty
//! - **Multi-objective optimization**: Primary/secondary fitness for complex objectives
//! - **Parallel evaluation**: Concurrent fitness assessment for performance
//! - **GPU Acceleration**: Mass parallel evaluation using compute shaders

use crate::{
    config::Config,
    gamestate::GameState,
    traits::Individual,
    tui::training::TrainingMessage,
    engines::gpu::GpuBatchEngine,
};
use rand::{prelude::*, rng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use tracing::{debug, info, trace, warn};

/// Represents a population of individuals (neural networks) for evolutionary training.
///
/// # Teaching Note: Population Structure Design
/// The population is the central data structure in genetic algorithms, maintaining:
/// - **Genotypes**: The actual individuals (neural network weights)
/// - **Phenotypes**: Fitness scores derived from individual performance
/// - **Parameters**: Configuration controlling evolutionary behavior
/// - **GPU Resources**: Batch processing engine for mass parallelization
///
/// # Multi-objective Fitness Encoding
/// Fitness scores use a packed `u64` format:
/// - **Upper 32 bits**: Primary objective (returns + shots)
/// - **Lower 32 bits**: Secondary objective (wins)
/// This enables **lexicographic sorting** where primary fitness dominates,
/// but secondary fitness breaks ties between equally fit individuals.
///
/// # Concurrency Design
/// Uses `AtomicU64` for fitness scores to enable:
/// - **Parallel fitness evaluation**: Multiple games can run simultaneously
/// - **Lock-free updates**: No contention between threads
/// - **Memory efficiency**: Direct atomic operations vs. mutex-protected data
///
/// # Memory Layout Optimization
/// - `Vec<I>` allows dynamic population sizing (vs. fixed arrays)
/// - Heap allocation acceptable since GA is not real-time critical
/// - Cache-friendly sequential access patterns for most operations
pub struct Population<I: Individual> {
    /// The population of neural networks (genomes/individuals)
    /// Each individual represents a complete neural network with all weights
    pub individuals: Vec<I>,
    
    /// Fitness scores for each individual using atomic operations for thread safety
    /// Packed format: [Primary: 32 bits][Secondary: 32 bits] 
    /// Enables multi-objective optimization with simple integer comparison
    pub fitness: Vec<AtomicU64>,
    
    /// Evolutionary algorithm configuration parameters
    /// Controls selection pressure, mutation rates, reproduction strategies, etc.
    pub config: Config,
    
    /// GPU batch processing engine for mass parallel evaluation
    /// Only initialized when GPU engine is selected
    gpu_batch_engine: Option<GpuBatchEngine>,
}

impl<I: Individual> Population<I> {
    /// Creates a new population with randomly initialized individuals.
    ///
    /// # Teaching Note: Population Initialization
    /// Initialization strategy affects evolutionary dynamics:
    /// - **Random initialization**: Explores full search space, no bias
    /// - **Diverse initialization**: Ensures genetic variety from start
    /// - **Population size**: Balance between diversity and computational cost
    /// - **GPU Preparation**: Initialize GPU resources if using GPU engine
    ///
    /// The `Individual::default()` creates random weights, providing
    /// **uniform random initialization** across the weight space.
    pub fn new(config: Config) -> Self {
        let pop_size = config.population_size;
        debug!("Creating population with size: {}", pop_size);
        
        // Initialize GPU batch engine if using GPU
        let gpu_batch_engine = if matches!(config.engine, crate::config::Engine::Gpu) {
            debug!("Initializing GPU batch engine...");
            match GpuBatchEngine::new(pop_size * 2) { // 2x for safety margin
                Ok(engine) => {
                    info!("GPU batch engine initialized for population size {}", pop_size);
                    Some(engine)
                }
                Err(e) => {
                    warn!("Failed to initialize GPU batch engine: {}. Falling back to CPU evaluation.", e);
                    warn!("GPU functionality is not available. Using CPU fallback mode.");
                    None
                }
            }
        } else {
            None
        };
        
        debug!("Creating {} individuals...", pop_size);
        let individuals: Vec<I> = (0..pop_size)
            .enumerate()
            .map(|(i, _)| {
                if i % 10 == 0 { debug!("Created {} individuals", i); }
                I::default()
            })
            .collect();
        debug!("All {} individuals created", pop_size);
        
        debug!("Creating fitness vector...");
        let fitness: Vec<AtomicU64> = (0..pop_size).map(|_| AtomicU64::new(0)).collect();
        debug!("Fitness vector created");
        
        debug!("Population creation complete");
        Self {
            individuals,
            fitness,
            config,
            gpu_batch_engine,
        }
    }

    /// Evaluates fitness through round-robin tournament (sequential version).
    ///
    /// # Teaching Note: Fitness Evaluation Strategies
    /// This implements **round-robin tournament evaluation**:
    /// - Every individual plays against every other individual
    /// - Fitness accumulates across all games (total performance)
    /// - More robust than single-game evaluation (reduces noise)
    /// - Computationally expensive: O(n²) games for population of size n
    ///
    /// # Algorithm Complexity:
    /// - **Time**: O(n² × game_length) where n = population_size
    /// - **Space**: O(n) for fitness storage
    /// - **Games**: n × (n-1) total matchups per generation
    ///
    /// # Alternative Evaluation Methods:
    /// - **Swiss Tournament**: O(n log n) games, less accurate
    /// - **Random Sampling**: O(n) games, very noisy but fast
    /// - **Elite vs All**: Only test against current champions
    pub fn evaluate_fitness(&mut self, _tx: &Option<mpsc::Sender<TrainingMessage>>) {
        trace!("Resetting fitness scores.");
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }

        let pop_size = self.config.population_size;
        trace!("Starting sequential full tournament (C++ equivalent).");

        // Round-robin tournament: every individual vs every other individual
        for i in 0..pop_size {
            for j in 0..pop_size {
                if i == j { continue; } // Skip self-play games
                
                // Run single game simulation between individuals i and j
                let mut game_state = GameState::new();
                let ((left_primary, left_secondary), (right_primary, right_secondary)) =
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);

                // Pack multi-objective fitness into single u64 value
                // Primary fitness (upper 32 bits) dominates, secondary (lower 32 bits) breaks ties
                let left_packed_score = ((left_primary as u64) << 32) | (left_secondary as u64);
                let right_packed_score = ((right_primary as u64) << 32) | (right_secondary as u64);
                
                // Accumulate fitness scores atomically (thread-safe)
                self.fitness[i].fetch_add(left_packed_score, Ordering::Relaxed);
                self.fitness[j].fetch_add(right_packed_score, Ordering::Relaxed);
            }
        }
        trace!("Sequential full tournament finished.");
    }

    /// Parallel version of fitness evaluation using Rayon for multi-core performance.
    ///
    /// # Teaching Note: Parallel Genetic Algorithms
    /// This demonstrates **embarrassingly parallel** fitness evaluation:
    /// - Each game simulation is independent (no shared state)
    /// - Perfect scaling with CPU cores (until memory bandwidth limits)
    /// - **Atomics** handle race conditions in fitness accumulation
    /// - **Work stealing** (Rayon) balances load across threads
    ///
    /// # Performance Characteristics:
    /// - **Linear speedup**: ~N× faster on N cores (ideal case)
    /// - **Memory bound**: Eventually limited by RAM bandwidth, not CPU
    /// - **Cache effects**: May suffer if population doesn't fit in cache
    ///
    /// # Synchronization Strategy:
    /// Uses lock-free atomic operations for maximum performance:
    /// - `fetch_add`: Atomically adds to fitness score
    /// - `Ordering::Relaxed`: Fastest memory ordering (sufficient here)
    /// - No locks = no thread contention or deadlock risks
    pub fn evaluate_fitness_concurrent(&mut self, tx: &Option<mpsc::Sender<TrainingMessage>>) {
        trace!("Resetting fitness scores.");
        self.fitness
            .par_iter()
            .for_each(|f| f.store(0, Ordering::Relaxed));

        let pop_size = self.config.population_size;
        
        // Generate all unique matchup pairs for round-robin tournament
        let pairs: Vec<(usize, usize)> = (0..pop_size)
            .flat_map(|i| (0..pop_size).filter(move |&j| i != j).map(move |j| (i, j)))
            .collect();

        trace!(
            total_games = pairs.len(),
            "Starting concurrent full tournament (C++ equivalent)."
        );

        if let Some(tx) = tx {
            // UI mode: send progress updates during evaluation
            pairs.par_iter().for_each_with(
                tx.clone(),
                |_thread_tx, &(i, j)| {
                    let mut game_state = GameState::new();
                    let ((left_primary, left_secondary), (right_primary, right_secondary)) =
                        game_state.simulate(
                            &self.individuals[i],
                            &self.individuals[j],
                            &self.config,
                        );

                    // Pack and accumulate fitness scores
                    let left_packed_score = ((left_primary as u64) << 32) | (left_secondary as u64);
                    let right_packed_score =
                        ((right_primary as u64) << 32) | (right_secondary as u64);

                    self.fitness[i].fetch_add(left_packed_score, Ordering::Relaxed);
                    self.fitness[j].fetch_add(right_packed_score, Ordering::Relaxed);
                },
            );
        } else {
            // CLI mode: no progress updates, maximum performance
            pairs.par_iter().for_each(|&(i, j)| {
                let mut game_state = GameState::new();
                let ((left_primary, left_secondary), (right_primary, right_secondary)) =
                    game_state.simulate(&self.individuals[i], &self.individuals[j], &self.config);

                // Pack and accumulate fitness scores
                let left_packed_score = ((left_primary as u64) << 32) | (left_secondary as u64);
                let right_packed_score = ((right_primary as u64) << 32) | (right_secondary as u64);

                self.fitness[i].fetch_add(left_packed_score, Ordering::Relaxed);
                self.fitness[j].fetch_add(right_packed_score, Ordering::Relaxed);
            });
        }
        trace!("Concurrent full tournament finished.");
    }

    /// Batch GPU fitness evaluation with fallback to CPU.
    ///
    /// # Teaching Note: GPU Computing with Graceful Degradation
    /// This function demonstrates production-ready GPU computing patterns:
    /// - **Primary GPU path**: Attempt high-performance GPU evaluation first
    /// - **Automatic fallback**: Fall back to CPU if GPU fails for any reason
    /// - **Error handling**: Graceful degradation ensures training always progresses
    /// - **Performance monitoring**: Track success/failure for optimization insights
    ///
    /// This pattern is essential in production ML systems where hardware varies
    /// and training must complete reliably regardless of available resources.
    pub fn evaluate_fitness_gpu_batch(&mut self, tx: &Option<mpsc::Sender<TrainingMessage>>) {
        trace!("Checking GPU batch engine availability...");
        
        // Check if GPU batch engine is available
        if self.gpu_batch_engine.is_none() {
            debug!("GPU batch engine not available, falling back to CPU evaluation");
            if self.config.concurrent {
                self.evaluate_fitness_concurrent(tx);
            } else {
                self.evaluate_fitness(tx);
            }
            return;
        }
        
        trace!("Resetting fitness scores for GPU batch evaluation.");
        for fitness in self.fitness.iter() {
            fitness.store(0, Ordering::Relaxed);
        }

        let pop_size = self.config.population_size;
        trace!("Starting GPU batch evaluation for {} individuals.", pop_size);

        // Calculate tournament size before borrowing
        let tournament_size = self.calculate_optimal_tournament_size();
        
        // Try GPU batch evaluation
        if let Some(batch_engine) = &mut self.gpu_batch_engine {
            match batch_engine.evaluate_population(&self.individuals, &self.config, tournament_size) {
                Ok(fitness_values) => {
                    // Success: update fitness scores from GPU results
                    for (i, &fitness_value) in fitness_values.iter().enumerate() {
                        if i < self.fitness.len() {
                            // Convert single fitness value to packed format (primary = fitness, secondary = 0)
                            let packed_fitness = ((fitness_value as u32 as u64) << 32) | 0u64;
                            self.fitness[i].store(packed_fitness, Ordering::Relaxed);
                        }
                    }
                    trace!("GPU batch evaluation completed successfully.");
                }
                Err(e) => {
                    warn!("GPU batch evaluation failed: {}. Falling back to CPU evaluation.", e);
                    // Fallback to CPU evaluation
                    if self.config.concurrent {
                        self.evaluate_fitness_concurrent(tx);
                    } else {
                        self.evaluate_fitness(tx);
                    }
                }
            }
        } else {
            // This shouldn't happen, but handle it gracefully
            warn!("GPU batch engine unexpectedly unavailable, using CPU fallback");
            if self.config.concurrent {
                self.evaluate_fitness_concurrent(tx);
            } else {
                self.evaluate_fitness(tx);
            }
        }
    }

    /// Calculates optimal tournament size for GPU workgroup efficiency.
    ///
    /// # Teaching Note: GPU Performance Optimization
    /// Tournament size affects GPU performance characteristics:
    /// - **Small tournaments**: More parallelism, less work per thread
    /// - **Large tournaments**: Less parallelism, more work per thread
    /// - **Optimal size**: Balance between parallelism and workgroup efficiency
    ///
    /// # Algorithm:
    /// Aims for GPU workgroup sizes that are multiples of warp/wavefront size (32/64)
    /// while ensuring sufficient work per thread for good GPU utilization.
    fn calculate_optimal_tournament_size(&self) -> usize {
        let population_size = self.individuals.len();
        
        // Target workgroup sizes that are efficient on most GPUs
        let target_workgroup_sizes = [64, 128, 256];
        
        // Find the largest tournament size that provides good parallelism
        for &workgroup_size in target_workgroup_sizes.iter().rev() {
            let tournament_size = population_size / workgroup_size;
            if tournament_size >= 4 && tournament_size <= 16 {
                return tournament_size;
            }
        }
        
        // Default fallback tournament size
        std::cmp::min(8, std::cmp::max(4, population_size / 32))
    }

    /// Selects elite individuals and returns their indices sorted by fitness.
    ///
    /// # Teaching Note: Selection in Genetic Algorithms
    /// Selection determines which individuals get to reproduce, implementing
    /// **"survival of the fittest"**. This function performs **fitness-based ranking**:
    ///
    /// ## Selection Pressure:
    /// High selection pressure (few elites) = fast convergence, risk of premature convergence
    /// Low selection pressure (many elites) = slow convergence, better exploration
    ///
    /// ## Multi-objective Fitness:
    /// The packed u64 fitness naturally implements **lexicographic ordering**:
    /// 1. Compare primary objectives first (upper 32 bits)
    /// 2. Use secondary objectives as tie-breakers (lower 32 bits)
    /// 3. Result: primary fitness dominates, secondary refines
    ///
    /// ## Alternative Selection Methods:
    /// - **Tournament Selection**: Random subgroups compete
    /// - **Roulette Wheel**: Probability proportional to fitness
    /// - **Rank-based**: Selection based on rank, not absolute fitness
    pub fn select_elites(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.config.population_size).collect();
        
        // Sort by packed fitness score (primary + secondary objectives)
        // Higher 32 bits (primary) automatically take precedence in comparison
        indices.sort_by_key(|&i| self.fitness[i].load(Ordering::Relaxed));
        indices.reverse(); // Descending order: highest fitness first
        
        indices
    }

    /// Calculates population diversity metrics to monitor convergence.
    ///
    /// # Teaching Note: Population Diversity
    /// Diversity monitoring prevents **premature convergence** - when the population
    /// becomes too similar too quickly, losing ability to explore the search space.
    ///
    /// ## Diversity Metrics:
    /// - **Genotypic diversity**: Variance in genome (weight) values
    /// - **Phenotypic diversity**: Variance in fitness scores
    /// - **Behavioral diversity**: Variance in actual behaviors/strategies
    ///
    /// ## Uses:
    /// - **Adaptive mutation**: Increase mutation when diversity is low
    /// - **Restart criteria**: Reinitialize when population converges
    /// - **Diagnosis**: Understand why evolution stagnates
    fn calculate_diversity_metrics(&self) -> (f32, f32) {
        let pop_size = self.config.population_size as f32;
        
        // Calculate fitness diversity (phenotypic)
        let fitness_values: Vec<f32> = self.fitness
            .iter()
            .map(|f| (f.load(Ordering::Relaxed) >> 32) as f32) // Primary fitness only
            .collect();
        
        let mean_fitness = fitness_values.iter().sum::<f32>() / pop_size;
        let fitness_variance = fitness_values
            .iter()
            .map(|&f| (f - mean_fitness).powi(2))
            .sum::<f32>() / pop_size;
        let fitness_diversity = fitness_variance.sqrt();
        
        // Calculate genetic diversity (genotypic) - sample of weight variance
        let sample_size = 100.min(crate::constants::TOTAL_WEIGHTS); // Sample weights for efficiency
        let mut weight_variances = Vec::new();
        
        for weight_idx in (0..crate::constants::TOTAL_WEIGHTS).step_by(crate::constants::TOTAL_WEIGHTS / sample_size) {
            let weight_values: Vec<f32> = self.individuals
                .iter()
                .map(|ind| ind.weights_as_slice()[weight_idx])
                .collect();
            
            let mean_weight = weight_values.iter().sum::<f32>() / pop_size;
            let weight_variance = weight_values
                .iter()
                .map(|&w| (w - mean_weight).powi(2))
                .sum::<f32>() / pop_size;
            weight_variances.push(weight_variance);
        }
        
        let genetic_diversity = weight_variances.iter().sum::<f32>() / weight_variances.len() as f32;
        
        (fitness_diversity, genetic_diversity.sqrt())
    }

    /// Creates the next generation using crossover and mutation of elite individuals.
    ///
    /// # Teaching Note: Genetic Operators and Reproduction Strategies
    /// This function implements the **reproduction phase** of the genetic algorithm,
    /// combining **selection**, **crossover**, and **mutation** to create offspring:
    ///
    /// ## C++ Equivalent Strategy:
    /// - **Survivor count**: √N individuals (balance between diversity and selection pressure)
    /// - **Elitism**: Top survivors carried forward unchanged (preserve best solutions)
    /// - **Crossover**: All pairwise combinations of survivors (systematic recombination)
    /// - **Mutation**: Random survivors mutated to fill remaining slots (exploration)
    ///
    /// ## Elite Crossover Strategy:
    /// - **Configurable elitism**: User-defined number of best individuals preserved
    /// - **Random mating**: Parents selected randomly from elite pool (higher diversity)
    /// - **Combined operators**: Each offspring undergoes both crossover and mutation
    ///
    /// # Genetic Operator Balance:
    /// - **Too much crossover**: Population converges quickly, may get stuck in local optima
    /// - **Too much mutation**: Population becomes random search, loses accumulated progress
    /// - **Balanced approach**: Crossover exploits known good areas, mutation explores new areas
    fn recombination_and_mutation(&mut self, sorted_indices: &[usize]) {
        let mut next_generation = Vec::with_capacity(self.config.population_size);
        let mut rng = rng();

        match self.config.reproduction_strategy {
            crate::config::ReproductionStrategy::CppEquivalent => {
                // C++ equivalent reproduction: systematic survivor-based approach
                let survivor_count = (self.config.population_size as f32).sqrt() as usize;

                // Phase 1: Elitism - preserve the best individuals unchanged
                // This ensures the best solutions are never lost due to genetic operators
                for i in 0..survivor_count {
                    next_generation.push(self.individuals[sorted_indices[i]].clone());
                }

                // Phase 2: Systematic crossover - all pairs of survivors produce offspring
                // This creates a thorough exploration of recombinations between good solutions
                let mut current_member_idx = survivor_count;
                'crossover: for i in 0..survivor_count {
                    for j in (i + 1)..survivor_count {
                        if current_member_idx >= self.config.population_size {
                            break 'crossover;
                        }
                        let parent1 = &self.individuals[sorted_indices[i]];
                        let parent2 = &self.individuals[sorted_indices[j]];
                        let offspring = parent1.crossover(parent2, &mut rng);
                        next_generation.push(offspring);
                        current_member_idx += 1;
                    }
                }

                // Phase 3: Mutation-based exploration - fill remaining slots with mutated survivors
                // This provides additional exploration beyond what crossover can achieve
                while current_member_idx < self.config.population_size {
                    for i in 0..survivor_count {
                        if current_member_idx >= self.config.population_size {
                            break;
                        }
                        let mut offspring = self.individuals[sorted_indices[i]].clone();
                        offspring.mutate(&mut rng, &self.config);
                        next_generation.push(offspring);
                        current_member_idx += 1;
                    }
                }
            }
            crate::config::ReproductionStrategy::EliteCrossover => {
                // Modern genetic algorithm: flexible elitism with random mating
                let elite_count = self.config.elite_count;

                // Phase 1: Elitism - carry over best individuals unchanged
                for i in 0..elite_count {
                    next_generation.push(self.individuals[sorted_indices[i]].clone());
                }

                // Phase 2: Random mating among elites with both crossover and mutation
                // This strategy provides higher genetic diversity than systematic mating
                for _ in elite_count..self.config.population_size {
                    // Randomly select two parents from elite pool
                    let parent1_idx = sorted_indices[rng.random_range(0..elite_count)];
                    let parent2_idx = sorted_indices[rng.random_range(0..elite_count)];
                    
                    // Apply crossover if probability check passes
                    let mut offspring = if rng.random::<f32>() < self.config.crossover_rate {
                        self.individuals[parent1_idx].crossover(&self.individuals[parent2_idx], &mut rng)
                    } else {
                        // No crossover: copy one parent (asexual reproduction)
                        self.individuals[parent1_idx].clone()
                    };
                    
                    // Apply mutation with configured probability and adaptive rate
                    let effective_mutation_rate = if self.config.adaptive_mutation {
                        self.calculate_adaptive_mutation_rate()
                    } else {
                        self.config.mutation_rate
                    };
                    
                    // Temporarily modify config for this mutation
                    let mut temp_config = self.config.clone();
                    temp_config.mutation_rate = effective_mutation_rate;
                    offspring.mutate(&mut rng, &temp_config);
                    
                    next_generation.push(offspring);
                }
            }
        }

        // Safety check: ensure exact population size (should never trigger with correct logic)
        next_generation.truncate(self.config.population_size);
        self.individuals = next_generation;
    }

    /// Calculates adaptive mutation rate based on population diversity.
    ///
    /// # Teaching Note: Adaptive Genetic Algorithms
    /// **Adaptive mutation** adjusts mutation rate dynamically based on population state:
    /// - **High diversity**: Lower mutation rate (population is exploring well)
    /// - **Low diversity**: Higher mutation rate (population needs more exploration)
    /// - **Benefits**: Automatic tuning, better balance of exploration vs exploitation
    ///
    /// # Algorithm:
    /// mutation_rate = min_rate + (max_rate - min_rate) × (1 - normalized_diversity)
    /// 
    /// Where normalized_diversity ∈ [0,1] based on genetic variance in population.
    fn calculate_adaptive_mutation_rate(&self) -> f32 {
        let (_, genetic_diversity) = self.calculate_diversity_metrics();
        
        // Normalize diversity to [0,1] range (estimated based on typical weight variance)
        let max_expected_diversity = 1.0; // Adjust based on empirical observations
        let normalized_diversity = (genetic_diversity / max_expected_diversity).clamp(0.0, 1.0);
        
        // Linear interpolation between min and max mutation rates
        // High diversity → low mutation rate, Low diversity → high mutation rate
        let range = self.config.max_mutation_rate - self.config.min_mutation_rate;
        self.config.min_mutation_rate + range * (1.0 - normalized_diversity)
    }

    /// Executes the complete evolutionary algorithm for the specified number of generations.
    ///
    /// # Teaching Note: Evolutionary Algorithm Main Loop
    /// This function orchestrates the complete **genetic algorithm lifecycle**:
    ///
    /// ## Evolution Cycle (repeated for each generation):
    /// 1. **Evaluation**: Measure fitness of all individuals through competition
    /// 2. **Selection**: Rank individuals by fitness, identify elites
    /// 3. **Reproduction**: Create next generation via crossover and mutation
    /// 4. **Assessment**: Monitor progress, check termination criteria
    ///
    /// ## Termination Criteria:
    /// - **Generation limit**: Stop after configured number of iterations
    /// - **Early stopping**: Halt if fitness plateaus (optional)
    /// - **User intervention**: Stop if UI channel closes
    /// - **Fitness threshold**: Stop if target fitness achieved (optional)
    ///
    /// ## Progress Monitoring:
    /// - **Real-time feedback**: Send updates to UI during evolution
    /// - **Comprehensive metrics**: Fitness statistics, training rates, convergence indicators
    /// - **Performance tracking**: Games per second, improvement rates, efficiency metrics
    ///
    /// # Returns
    /// The best individual found during the entire evolutionary process.
    /// This represents the "champion" neural network with highest fitness.
    pub fn evolve(&mut self, tx: Option<mpsc::Sender<TrainingMessage>>) -> I {
        debug!(
            generations = self.config.generations,
            "Starting evolution loop."
        );
        
        let start_time = std::time::Instant::now();
        let mut fitness_history: Vec<f32> = Vec::new();
        let mut total_matches_simulated = 0u64;
        let mut best_fitness_ever = 0.0f32;
        let mut generations_without_improvement = 0u32;
        
        for gen in 0..self.config.generations {
            debug!(generation = gen + 1, "Starting generation.");

            // Phase 1: Fitness Evaluation
            // All individuals compete to determine their relative performance
            trace!("Evaluating fitness...");
            let use_gpu = matches!(self.config.engine, crate::config::Engine::Gpu);
            if use_gpu {
                self.evaluate_fitness_gpu_batch(&tx);
            } else if self.config.concurrent {
                self.evaluate_fitness_concurrent(&tx);
            } else {
                self.evaluate_fitness(&tx);
            }
            trace!("Fitness evaluation complete.");

            // Phase 2: Selection and Ranking
            // Identify the best individuals for reproduction
            trace!("Selecting elites...");
            let sorted_indices = self.select_elites();
            trace!("Elite selection complete.");

            // Phase 3: Performance Analysis and Reporting
            let best_fitness_packed = self.fitness[sorted_indices[0]].load(Ordering::Relaxed);
            let worst_fitness_packed = self.fitness
                [sorted_indices[self.config.population_size - 1]]
                .load(Ordering::Relaxed);

            // Extract primary fitness scores for reporting
            let best_fitness = (best_fitness_packed >> 32) as f32;
            let worst_fitness = (worst_fitness_packed >> 32) as f32;

            // Calculate population fitness statistics
            let average_fitness = self
                .fitness
                .iter()
                .map(|f| (f.load(Ordering::Relaxed) >> 32) as u32) // Primary score only
                .sum::<u32>() as f32
                / self.config.population_size as f32;

            let best_individual = &self.individuals[sorted_indices[0]];

            // Calculate training performance metrics
            let total_matches_this_gen = (self.config.population_size * (self.config.population_size - 1)) as u64;
            total_matches_simulated += total_matches_this_gen;
            let elapsed_time = start_time.elapsed().as_secs_f32();
            let training_rate = if elapsed_time > 0.0 { (gen + 1) as f32 / elapsed_time } else { 0.0 };
            
            // Track improvement and convergence
            let gen_usize = gen as usize;
            let improvement_rate = if gen > 0 && elapsed_time > 0.0 {
                let prev_best = fitness_history.get(gen_usize.saturating_sub(1)).copied().unwrap_or(0.0);
                let improvement = (best_fitness - prev_best).max(0.0);
                let gen_duration = elapsed_time / (gen + 1) as f32;
                if gen_duration > 0.0 { improvement / gen_duration } else { 0.0 }
            } else {
                0.0
            };
            
            // Update fitness history and convergence tracking
            if fitness_history.len() <= gen_usize {
                fitness_history.resize(gen_usize + 1, 0.0);
            }
            fitness_history[gen_usize] = best_fitness;
            
            // Check for improvement and early stopping
            if best_fitness > best_fitness_ever {
                best_fitness_ever = best_fitness;
                generations_without_improvement = 0;
            } else {
                generations_without_improvement += 1;
            }
            
            // Early stopping check
            if let Some(patience) = self.config.early_stopping_patience {
                if generations_without_improvement >= patience {
                    info!("Early stopping triggered: no improvement for {} generations", patience);
                    if let Some(tx) = &tx {
                        let _ = tx.send(TrainingMessage::EarlyStopping { 
                            final_generation: (gen + 1) as usize,
                            best_fitness: best_fitness_ever,
                        });
                    }
                    return self.individuals[sorted_indices[0]].clone();
                }
            }

            // Theoretical maximum score analysis (for progress visualization)
            const MAX_POSSIBLE_SCORE: u32 = 50; // Based on game mechanics analysis
            
            // Phase 4: Progress Reporting
            let progress_message = TrainingMessage::Progress {
                generation: (gen + 1) as usize,
                best_fitness: best_fitness as f32,
                genome_weights: best_individual.weights_as_slice().to_vec(),
                total_matches_simulated,
                training_rate,
                improvement_rate,
            };

            if let Some(tx) = &tx {
                if tx.send(progress_message).is_err() {
                    // UI disconnected: return best individual found so far
                    debug!(
                        generation = gen + 1,
                        "Evolution stopped early: UI channel closed."
                    );
                    return self.individuals[sorted_indices[0]].clone();
                }
            } else {
                // CLI mode: comprehensive progress display
                let diversity_metrics = if self.config.track_diversity {
                    self.calculate_diversity_metrics()
                } else {
                    (0.0, 0.0)
                };
                
                info!(
                    "Gen: {:3} | Best: {:7.2} | Avg: {:7.2} | Worst: {:7.2} | Matches: {:8} | Rate: {:5.2} gen/s | Improve: {:6.3} fit/s | Diversity: ({:.3}, {:.3}) | Max: {}",
                    gen + 1,
                    best_fitness,
                    average_fitness,
                    worst_fitness,
                    total_matches_simulated,
                    training_rate,
                    improvement_rate,
                    diversity_metrics.0,
                    diversity_metrics.1,
                    MAX_POSSIBLE_SCORE
                );
            }

            // Phase 5: Reproduction (create next generation)
            trace!("Performing recombination and mutation...");
            self.recombination_and_mutation(&sorted_indices);
            trace!("Recombination and mutation complete.");
            debug!(generation = gen + 1, "Generation complete.");
        }

        // Evolution complete: return the champion individual
        debug!("Evolution loop finished. Selecting final best individual.");
        let sorted_indices = self.select_elites();
        self.individuals[sorted_indices[0]].clone()
    }
}
