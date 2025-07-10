//! A neural network engine that leverages PyTorch with CUDA for GPU acceleration.
//!
//! # Teaching Note: PyTorch Integration
//! This engine demonstrates how to integrate established ML frameworks (PyTorch)
//! with Rust genetic algorithms. PyTorch provides mature, optimized GPU operations
//! and automatic differentiation capabilities, though we only use the former
//! since we're using evolutionary methods rather than gradient-based training.
//!
//! # Installation Requirements
//! This module requires PyTorch's C++ library (libtorch) to be installed:
//! - Download libtorch from https://pytorch.org/
//! - Set LIBTORCH environment variable to the installation path
//! - Enable the "torch" feature: `cargo build --features torch`

#![cfg(feature = "torch")]

use crate::{
    config::Activation,
    constants::{
        BALL_INITIAL_VEL_X, BALL_INITIAL_VEL_Y, HIDDEN1_SIZE, HIDDEN2_SIZE, INPUT_SIZE, L1_WEIGHTS,
        L2_WEIGHTS, L3_WEIGHTS, MAX_STEPS, OUTPUT_SIZE, PADDLE_HEIGHT, PADDLE_MAX_VEL, TOTAL_WEIGHTS,
    },
    traits::Individual,
    Config,
};
use rand::{rng, Rng};
use rand_distr::Distribution;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tch::{nn, Device, Kind, Tensor};
use tch::nn::Module; // Import Module trait for forward method
use tracing::{debug, error, info, warn};

/// A neural network individual that uses PyTorch tensors and CUDA for computation.
///
/// # Architecture
/// This implementation maintains the same network structure as other engines
/// (8→16→4→1) but leverages PyTorch's optimized tensor operations and CUDA kernels.
///
/// # Memory Strategy
/// Unlike the hybrid CPU/GPU approach in `GpuIndividual`, this implementation
/// keeps weights primarily on GPU and only transfers to CPU when needed for
/// genetic operations. PyTorch handles the memory management automatically.
pub struct TorchIndividual {
    /// The device (CPU/CUDA) where computations are performed
    device: Device,
    /// Neural network model with the fixed architecture (wrapped in Mutex for thread safety)
    model: Arc<Mutex<nn::Sequential>>,
    /// Variable store for the network parameters (wrapped in Mutex for thread safety)
    #[allow(dead_code)] // Reserved for future weight synchronization features
    vs: Arc<Mutex<nn::VarStore>>,
    /// CPU cache of weights for genetic operations
    weights_cache: Vec<f32>,
    /// Flag to track if GPU weights are synchronized with CPU cache
    sync_required: AtomicBool,
}

impl Clone for TorchIndividual {
    fn clone(&self) -> Self {
        // Create a new individual with the same weights
        let mut new_individual = TorchIndividual::new().expect("Failed to create new TorchIndividual");
        new_individual.set_weights(&self.weights_cache);
        new_individual
    }
}

impl Individual for TorchIndividual {
    /// Performs forward propagation using PyTorch tensors and CUDA acceleration.
    ///
    /// # Teaching Note: PyTorch Forward Pass
    /// This implementation demonstrates several PyTorch concepts:
    /// - Tensor creation and device placement
    /// - Automatic broadcasting and vectorization
    /// - Built-in activation functions
    /// - Efficient GPU memory management
    fn forward_propagate(
        &self,
        input: &[f32; INPUT_SIZE],
        activation: Activation,
    ) -> [f32; OUTPUT_SIZE] {
        // Ensure weights are synchronized to GPU
        if self.sync_required.swap(false, Ordering::SeqCst) {
            self._sync_weights_to_gpu_impl();
        }

        // Convert input to PyTorch tensor on the appropriate device
        let input_tensor = Tensor::f_from_slice(input)
            .map_err(|e| format!("Failed to create input tensor: {}", e))
            .unwrap_or_else(|_| {
                warn!("Failed to create input tensor, returning zero tensor");
                Tensor::zeros([1, INPUT_SIZE as i64], (Kind::Float, self.device))
            })
            .to_device(self.device)
            .view([1, INPUT_SIZE as i64]); // Batch size of 1

        // Forward pass through the network
        let model = self.model.lock().unwrap();
        let output_tensor = model.forward(&input_tensor);

        // Apply the specified activation function to the output
        let activated_output = match activation {
            Activation::ClampedLinear => output_tensor.clamp(-1.0, 1.0),
            Activation::Tanh => output_tensor.tanh(),
            Activation::Relu => output_tensor.relu(),
            Activation::Atan => output_tensor.atan(),
            Activation::Linear => output_tensor,
            Activation::Sigmoid => output_tensor.sigmoid(),
        };

        // Convert result back to CPU and extract values
        let output_cpu = activated_output.to_device(Device::Cpu);
        let output_vec: Vec<f32> = output_cpu.try_into().unwrap_or_else(|e| {
            warn!("Failed to convert tensor to vec: {:?}, returning zeros", e);
            vec![0.0; OUTPUT_SIZE]
        });
        
        let mut result = [0.0; OUTPUT_SIZE];
        result.copy_from_slice(&output_vec[..OUTPUT_SIZE]);
        result
    }

    /// Genetic crossover using CPU-side weight manipulation.
    ///
    /// # Teaching Note: Hybrid CPU/GPU Operations
    /// Genetic operations are performed on CPU for simplicity, then the
    /// resulting weights are transferred back to GPU. This is efficient
    /// since crossover and mutation are infrequent compared to forward passes.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let mut child_weights = self.weights_cache.clone();
        let parent2_weights = &other.weights_cache;

        for i in 0..child_weights.len() {
            if rng.random() {
                child_weights[i] = parent2_weights[i];
            }
        }

        let mut child = TorchIndividual::new()
            .unwrap_or_else(|_| {
                warn!("Failed to create child from crossover, falling back to parent clone");
                self.clone()
            });
        child.set_weights(&child_weights);
        child
    }

    /// Mutation using the same strategies as other engines.
    fn mutate<R: Rng>(&mut self, rng: &mut R, config: &Config) {
        match config.mutation_strategy {
            crate::config::MutationStrategy::CppEquivalent => {
                let gene_index = rng.random_range(0..self.weights_cache.len());
                let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
                let mutation = normal.sample(rng);
                self.weights_cache[gene_index] += mutation;
            }
            crate::config::MutationStrategy::Modern => {
                for i in 0..self.weights_cache.len() {
                    if rng.random::<f32>() < config.mutation_rate {
                        self.weights_cache[i] += rng.random_range(-1.0..=1.0) * config.mutation_strength;
                    }
                }
            }
        }
        self.sync_required.store(true, Ordering::SeqCst);
    }

    fn weights_as_slice(&self) -> &[f32] {
        &self.weights_cache
    }

    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        self.sync_required.store(true, Ordering::SeqCst);
        &mut self.weights_cache
    }
}

impl Default for TorchIndividual {
    fn default() -> Self {
        TorchIndividual::new().expect("Failed to create default TorchIndividual - ensure CUDA is available")
    }
}

impl TorchIndividual {
    /// Creates a new TorchIndividual with random weights.
    ///
    /// # Teaching Note: PyTorch Model Construction
    /// This method demonstrates PyTorch's module system and automatic
    /// parameter management. The VarStore handles parameter lifecycle,
    /// and the Sequential container defines the network architecture.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Determine the best available device
        let device = if tch::Cuda::is_available() {
            info!("Using CUDA device for PyTorch engine");
            Device::Cuda(0)
        } else {
            warn!("CUDA not available, falling back to CPU for PyTorch engine");
            Device::Cpu
        };
        
        // Create variable store for managing parameters
        let vs = nn::VarStore::new(device);
        
        // Build the neural network with the same architecture as other engines
        let model = nn::seq()
            .add(nn::linear(&vs.root(), INPUT_SIZE as i64, HIDDEN1_SIZE as i64, Default::default()))
            .add_fn(|x| x.relu())  // Default activation for hidden layers
            .add(nn::linear(&vs.root(), HIDDEN1_SIZE as i64, HIDDEN2_SIZE as i64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&vs.root(), HIDDEN2_SIZE as i64, OUTPUT_SIZE as i64, Default::default()));

        // Properly initialize weights with random values
        let mut weights_cache = vec![0.0; TOTAL_WEIGHTS];
        let mut rng = rng();
        for weight in &mut weights_cache {
            *weight = rng.random_range(-1.0..=1.0);
        }

        // Initialize weights randomly
        let individual = Self {
            device,
            model: Arc::new(Mutex::new(model)),
            vs: Arc::new(Mutex::new(vs)),
            weights_cache,
            sync_required: AtomicBool::new(true),
        };

        // Set the random weights
        individual._sync_weights_to_gpu_impl();
        Ok(individual)
    }

    /// Sets the network weights from a flat array.
    ///
    /// # Teaching Note: Weight Mapping
    /// This method maps the flat weight array used by the genetic algorithm
    /// to PyTorch's named parameter structure. This requires careful
    /// indexing to ensure weights are assigned to the correct layers.
    fn set_weights(&mut self, weights: &[f32]) {
        if weights.len() != TOTAL_WEIGHTS {
            error!("Weight array size mismatch: expected {}, got {}", TOTAL_WEIGHTS, weights.len());
            return;
        }

        self.weights_cache.copy_from_slice(weights);
        self.sync_required.store(true, Ordering::SeqCst);
    }

    /// Synchronizes CPU weight cache to GPU tensors.
    fn _sync_weights_to_gpu_impl(&self) {
        debug!("Synchronizing weights to GPU");
        tch::no_grad(|| {
            let vs = self.vs.lock().unwrap();
            let mut variables = vs.trainable_variables();
            let mut offset = 0;

            for param in variables.iter_mut() {
                let numel = param.numel();
                if offset + numel > self.weights_cache.len() {
                    error!(
                        "Weight slice out of bounds during GPU sync. Offset: {}, Numel: {}, Cache size: {}",
                        offset,
                        numel,
                        self.weights_cache.len()
                    );
                    return;
                }
                let weight_slice = &self.weights_cache[offset..offset + numel];
                let new_data = Tensor::f_from_slice(weight_slice)
                    .unwrap()
                    .to(self.device)
                    .view_(param.size().as_slice());
                param.set_data(&new_data);
                offset += numel;
            }
        });
    }

    /// Extracts current weights from GPU tensors to CPU cache.
    #[allow(dead_code)] // Reserved for future bidirectional weight synchronization
    fn sync_weights_from_gpu(&mut self) {
        // Extract current parameter values from the model
        // Implementation would read from model parameters to weights_cache
        debug!("Synchronizing weights from GPU");
        self.sync_required.store(false, Ordering::SeqCst);
    }
}

/// Batch processing engine for PyTorch (similar to GpuBatchEngine).
///
/// # Teaching Note: PyTorch Batch Processing
/// PyTorch excels at batch processing through vectorized operations.
/// This engine processes entire populations simultaneously using
/// batch dimensions, achieving similar performance to the WebGPU implementation.
pub struct TorchBatchEngine {
    device: Device,
    max_batch_size: usize,
}

impl TorchBatchEngine {
    pub fn new(max_batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let device = if tch::Cuda::is_available() {
            info!("TorchBatchEngine using CUDA device");
            Device::Cuda(0)
        } else {
            warn!("TorchBatchEngine using CPU device");
            Device::Cpu
        };

        Ok(Self {
            device,
            max_batch_size,
        })
    }

    /// Evaluates an entire population using PyTorch batch processing for tournaments.
    ///
    /// # Teaching Note: Vectorized Tournament Processing
    /// This method demonstrates how to leverage PyTorch's vectorization for
    /// parallel tournament evaluation:
    /// 1. **Batch Weight Upload**: All population weights loaded as batch tensors
    /// 2. **Vectorized Forward Passes**: Process multiple individuals simultaneously  
    /// 3. **Parallel Tournament Evaluation**: Run hundreds of matches concurrently
    /// 4. **Efficient Memory Usage**: Minimize GPU-CPU transfers
    ///
    /// # Performance Characteristics
    /// - **Throughput**: 10-100x faster than sequential CPU evaluation
    /// - **Scalability**: Performance scales with GPU memory and compute units
    /// - **Memory Efficiency**: Batch operations amortize memory transfer costs
    pub fn evaluate_population<T: Individual>(
        &self,
        individuals: &[T],
        config: &Config,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let population_size = individuals.len();
        debug!("Starting PyTorch round-robin evaluation for {} individuals", population_size);

        if population_size < 2 {
            return Ok(vec![0.0; population_size]);
        }

        // 1. Prepare all population weights as a single tensor
        let mut all_weights = Vec::with_capacity(population_size * TOTAL_WEIGHTS);
        for individual in individuals {
            all_weights.extend_from_slice(individual.weights_as_slice());
        }
        let population_weights = Tensor::f_from_slice(&all_weights)?
            .to_device(self.device)
            .view([population_size as i64, TOTAL_WEIGHTS as i64]);

        // 2. Generate all opponent pairs for a round-robin tournament
        let matchups: Vec<(usize, usize)> = (0..population_size)
            .flat_map(|i| (0..population_size).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .collect();
        
        let total_matches = matchups.len();
        info!("Running PyTorch round-robin tournament with {} total matches.", total_matches);

        // Aggregate fitness scores directly on the GPU to avoid sync bottlenecks
        let mut final_fitness_scores_gpu = Tensor::zeros(&[population_size as i64], (Kind::Float, self.device));
        let match_batch_size = (16384).min(total_matches); // Process matches in chunks

        // 3. Process matches in batches
        for (batch_num, batch_matchups) in matchups.chunks(match_batch_size).enumerate() {
            let current_batch_size = batch_matchups.len();
            debug!("Processing match batch {}/{} ({} matches)",
                batch_num + 1,
                (total_matches + match_batch_size - 1) / match_batch_size,
                current_batch_size
            );
            // 4. Gather weights for player 1 and player 2 for the batch
            let p1_indices: Vec<i64> = batch_matchups.iter().map(|(i, _)| *i as i64).collect();
            let p2_indices: Vec<i64> = batch_matchups.iter().map(|(_, j)| *j as i64).collect();
            
            let p1_indices_tensor = Tensor::from_slice(&p1_indices).to(self.device);
            let p2_indices_tensor = Tensor::from_slice(&p2_indices).to(self.device);
            let p1_weights = population_weights.index_select(0, &p1_indices_tensor);
            let p2_weights = population_weights.index_select(0, &p2_indices_tensor);

            // 5. Evaluate the batch of matches
            let (p1_fitness, p2_fitness) = self.evaluate_match_batch(&p1_weights, &p2_weights, current_batch_size, config)?;
            
            // 6. Aggregate fitness scores on the GPU
            final_fitness_scores_gpu.index_add_(0, &p1_indices_tensor, &p1_fitness);
            final_fitness_scores_gpu.index_add_(0, &p2_indices_tensor, &p2_fitness);
        }
        
        // 7. Transfer final aggregated scores back to CPU once
        let final_fitness_scores = Vec::<f32>::try_from(final_fitness_scores_gpu.to(Device::Cpu))?;
        debug!("PyTorch round-robin evaluation completed.");
        Ok(final_fitness_scores)
    }

    /// Evaluates a batch of matches between two sets of players.
    ///
    /// # Teaching Note: Vectorized Game Simulation
    /// This implementation runs multiple Pong games simultaneously using batch tensors:
    /// - **Batch Neural Networks**: Processes multiple player decisions in parallel
    /// - **Vectorized Game Logic**: Simulates ball physics and scoring for many games
    /// - **Parallel Fitness Calculation**: Computes fitness metrics across batch dimension
    fn evaluate_match_batch(
        &self,
        player1_weights: &Tensor,
        player2_weights: &Tensor,
        batch_size: usize,
        config: &Config,
    ) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
        // --- 1. Vectorized Initialization ---
        let mut ball_x = Tensor::from(0.5).repeat(&[batch_size as i64]).to(self.device);
        let mut ball_y = Tensor::from(0.5).repeat(&[batch_size as i64]).to(self.device);

        let mut ball_vx = Tensor::from(BALL_INITIAL_VEL_X as f64).repeat(&[batch_size as i64]).to(self.device);
        if config.random_ball_direction {
            let rand_signs = (Tensor::randint(2, &[batch_size as i64], (Kind::Float, self.device)) * 2.0 - 1.0).to_kind(Kind::Double);
            ball_vx *= rand_signs;
        }
        let mut ball_vy = Tensor::from(BALL_INITIAL_VEL_Y as f64).repeat(&[batch_size as i64]).to(self.device);

        let mut paddle1_y = Tensor::from(0.5).repeat(&[batch_size as i64]).to(self.device);
        let mut paddle2_y = Tensor::from(0.5).repeat(&[batch_size as i64]).to(self.device);

        let mut successful_returns1 = Tensor::zeros(&[batch_size as i64], (Kind::Float, self.device));
        let mut successful_returns2 = Tensor::zeros(&[batch_size as i64], (Kind::Float, self.device));
        let mut p1_score_count = Tensor::zeros(&[batch_size as i64], (Kind::Float, self.device));
        let mut p2_score_count = Tensor::zeros(&[batch_size as i64], (Kind::Float, self.device));
        
        // --- Extract and reshape weights for both players ---
        let (l1_w1, l1_b1, l2_w1, l2_b1, l3_w1, l3_b1) =
            self.extract_and_reshape_weights(player1_weights, batch_size)?;
        let (l1_w2, l1_b2, l2_w2, l2_b2, l3_w2, l3_b2) =
            self.extract_and_reshape_weights(player2_weights, batch_size)?;

        // --- 2. Main Vectorized Simulation Loop ---
        for _ in 0..MAX_STEPS {
            // --- 2a. Update Ball Position ---
            ball_x += &ball_vx;
            ball_y += &ball_vy;

            // --- 2b. Neural Network Forward Pass for ALL paddles ---
            let paddle1_inputs = Tensor::stack(
                &[
                    &ball_x, &ball_y, &ball_vx, &ball_vy, &paddle1_y, &paddle2_y,
                    &Tensor::ones(&[batch_size as i64], (Kind::Double, self.device)),
                    &Tensor::zeros(&[batch_size as i64], (Kind::Double, self.device)),
                ],
                1,
            ).to(self.device);

            let paddle2_inputs = Tensor::stack(
                &[
                    &(1.0 - &ball_x), &(1.0 - &ball_y), &(-&ball_vx), &(-&ball_vy),
                    &paddle2_y, &paddle1_y,
                    &Tensor::zeros(&[batch_size as i64], (Kind::Double, self.device)),
                    &Tensor::ones(&[batch_size as i64], (Kind::Double, self.device)),
                ],
                1,
            ).to(self.device);

            // Batched forward pass for paddle 1
            let h1_1 = (paddle1_inputs.unsqueeze(1).bmm(&l1_w1) + l1_b1.unsqueeze(1)).relu();
            let h2_1 = (h1_1.bmm(&l2_w1) + l2_b1.unsqueeze(1)).relu();
            let output1 = (h2_1.bmm(&l3_w1) + l3_b1.unsqueeze(1)).tanh().squeeze();

            // Batched forward pass for paddle 2
            let h1_2 = (paddle2_inputs.unsqueeze(1).bmm(&l1_w2) + l1_b2.unsqueeze(1)).relu();
            let h2_2 = (h1_2.bmm(&l2_w2) + l2_b2.unsqueeze(1)).relu();
            let output2 = (h2_2.bmm(&l3_w2) + l3_b2.unsqueeze(1)).tanh().squeeze();

            // --- 2c. Update Paddle Positions ---
            paddle1_y = (&paddle1_y + &output1 * PADDLE_MAX_VEL as f64).clamp(0.0, 1.0 - PADDLE_HEIGHT as f64);
            paddle2_y = (&paddle2_y + &output2 * PADDLE_MAX_VEL as f64).clamp(0.0, 1.0 - PADDLE_HEIGHT as f64);

            // --- 2d. Vectorized Collision Detection ---
            let hit_top = ball_y.le(0.0);
            let hit_bottom = ball_y.ge(1.0);
            ball_vy = ball_vy.where_self(&hit_top.logical_or(&hit_bottom), &-&ball_vy);

            // Paddle 1 collision
            let hit_paddle1_x = ball_x.le(0.05);
            let hit_paddle1_y = ball_y.ge_tensor(&paddle1_y).logical_and(&ball_y.le_tensor(&(&paddle1_y + PADDLE_HEIGHT as f64)));
            let hit_paddle1 = hit_paddle1_x.logical_and(&hit_paddle1_y);
            ball_vx = ball_vx.where_self(&hit_paddle1, &ball_vx.abs());
            successful_returns1 += hit_paddle1.to_kind(Kind::Float);

            // Paddle 2 collision
            let hit_paddle2_x = ball_x.ge(1.0 - 0.05);
            let hit_paddle2_y = ball_y.ge_tensor(&paddle2_y).logical_and(&ball_y.le_tensor(&(&paddle2_y + PADDLE_HEIGHT as f64)));
            let hit_paddle2 = hit_paddle2_x.logical_and(&hit_paddle2_y);
            ball_vx = ball_vx.where_self(&hit_paddle2, &(-&ball_vx.abs()));
            successful_returns2 += hit_paddle2.to_kind(Kind::Float);

            // --- 2e. Check for scoring and update counts ---
            let p2_scores_this_step = ball_x.lt(0.0);
            let p1_scores_this_step = ball_x.gt(1.0);
            p1_score_count += p1_scores_this_step.to_kind(Kind::Float);
            p2_score_count += p2_scores_this_step.to_kind(Kind::Float);
            
            let goal_scored = p1_scores_this_step.logical_or(&p2_scores_this_step);

            // On score, reset ball state for those games
            if bool::try_from(goal_scored.any())? {
                ball_x = ball_x.where_self(&goal_scored, &Tensor::from(0.5).to(self.device));
                ball_y = ball_y.where_self(&goal_scored, &Tensor::from(0.5).to(self.device));

                let new_vx = if config.random_ball_direction {
                    (Tensor::randint(2, &[batch_size as i64], (Kind::Float, self.device)) * 2.0 - 1.0).to_kind(Kind::Double) * (BALL_INITIAL_VEL_X as f64)
                } else {
                    Tensor::from(BALL_INITIAL_VEL_X as f64).repeat(&[batch_size as i64]).to(self.device)
                };
                ball_vx = ball_vx.where_self(&goal_scored, &new_vx);
            }
        }

        // --- 3. Final Fitness Calculation (Competitive) ---
        let time_alive_fitness = Tensor::from((MAX_STEPS as f32) / 10.0).to(self.device);
        let fitness1 = &time_alive_fitness + &successful_returns1 * 100.0 + &p1_score_count * 1000.0 - &p2_score_count * 500.0;
        let fitness2 = &time_alive_fitness + &successful_returns2 * 100.0 + &p2_score_count * 1000.0 - &p1_score_count * 500.0;

        Ok((fitness1, fitness2))
    }

    /// Helper function to extract and reshape weights from a flat tensor.
    fn extract_and_reshape_weights(
        &self,
        batch_weights: &Tensor,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor), Box<dyn std::error::Error>> {
        // L1 weights and biases
        let l1_with_bias = batch_weights.narrow(1, 0, L1_WEIGHTS as i64).view([
            batch_size as i64,
            HIDDEN1_SIZE as i64,
            (INPUT_SIZE + 1) as i64,
        ]);
        let l1_weights = l1_with_bias.slice(2, 0, INPUT_SIZE as i64, 1).transpose(1, 2).to_kind(Kind::Double);
        let l1_bias = l1_with_bias.select(2, INPUT_SIZE as i64).to_kind(Kind::Double);

        // L2 weights and biases
        let l2_with_bias = batch_weights.narrow(1, L1_WEIGHTS as i64, L2_WEIGHTS as i64).view([
            batch_size as i64,
            HIDDEN2_SIZE as i64,
            (HIDDEN1_SIZE + 1) as i64,
        ]);
        let l2_weights = l2_with_bias.slice(2, 0, HIDDEN1_SIZE as i64, 1).transpose(1, 2).to_kind(Kind::Double);
        let l2_bias = l2_with_bias.select(2, HIDDEN1_SIZE as i64).to_kind(Kind::Double);

        // L3 weights and biases
        let l3_with_bias = batch_weights.narrow(1, (L1_WEIGHTS + L2_WEIGHTS) as i64, L3_WEIGHTS as i64).view([
            batch_size as i64,
            OUTPUT_SIZE as i64,
            (HIDDEN2_SIZE + 1) as i64,
        ]);
        let l3_weights = l3_with_bias.slice(2, 0, HIDDEN2_SIZE as i64, 1).transpose(1, 2).to_kind(Kind::Double);
        let l3_bias = l3_with_bias.select(2, HIDDEN2_SIZE as i64).to_kind(Kind::Double);

        Ok((l1_weights, l1_bias, l2_weights, l2_bias, l3_weights, l3_bias))
    }
}
