//! An experimental neural network engine that leverages the GPU for massively parallel
//! tournament evaluation using WGSL shaders.
//!
//! # Teaching Note: GPU Computing in Machine Learning
//! GPUs excel at parallel computation, making them ideal for neural network operations.
//! While this implementation focuses on forward propagation, modern ML frameworks
//! like TensorFlow and PyTorch use similar principles for both forward and backward
//! passes, training massive networks with millions of parameters in parallel.
//!
//! # New: Mass Parallel Processing Architecture
//! This engine now supports true GPU mass parallelization:
//! - **Batch Processing**: Evaluate entire populations simultaneously
//! - **Tournament Evaluation**: Run tournaments entirely on GPU
//! - **Memory Optimization**: Efficient GPU memory management for large populations
//! - **Scalability**: Performance scales linearly with GPU compute units

use crate::{config::Activation, constants::*, traits::Individual, Config};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::Lazy;
use pollster::block_on;
use rand::Rng;
use rand_distr::Distribution;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use tracing::{error, warn, info, debug};

// A global, lazily-initialized GPU context, shared across all GpuIndividuals.
static GPU_CONTEXT: Lazy<Option<GpuContext>> = Lazy::new(|| {
    match GpuContext::new() {
        Ok(context) => Some(context),
        Err(e) => {
            error!("Failed to initialize GPU context: {}", e);
            None
        }
    }
});

/// Holds the WGPU device, queue, and the pre-compiled compute pipelines.
///
/// # Teaching Note: Enhanced GPU Context for Batch Processing
/// The context now includes both single-individual and batch processing pipelines.
/// This allows for flexible usage depending on the scenario:
/// - Single processing for testing and debugging
/// - Batch processing for high-performance population evaluation
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,           // Single individual pipeline
    batch_pipeline: wgpu::ComputePipeline,     // Batch tournament pipeline
    max_compute_units: u32,                    // GPU compute capability info
}

/// Uniform data passed from the CPU to the GPU shader.
/// 
/// # Teaching Note: GPU-CPU Data Transfer
/// This struct represents data that changes between shader invocations but remains
/// constant during a single compute dispatch. The `#[repr(C)]` ensures the memory
/// layout matches what the GPU expects, while `Pod` and `Zeroable` from bytemuck
/// allow safe casting to bytes for GPU transfer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuConfig {
    /// Activation function selector - MUST match the WGSL shader:
    /// 0: ClampedLinear, 1: Tanh, 2: ReLU, 3: Atan, 4: Linear, 5: Sigmoid
    activation_type: u32,
}

/// Batch configuration for mass parallel processing
///
/// # Teaching Note: Round-Robin Processing Configuration
/// This structure contains all the parameters needed for GPU round-robin processing:
/// - Population management (size, total matches)
/// - Algorithm configuration (activation, fitness)
/// - Performance tuning (random seed for reproducibility)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchConfig {
    population_size: u32,     // Number of individuals in population
    total_matches: u32,       // Total matches in round-robin
    activation_type: u32,     // Activation function (0-5)
    random_seed: u32,         // Random seed for reproducibility
    fitness_function: u32,    // Fitness function type (0-2)
    workgroup_offset: u32,    // Offset for chunked dispatch
    random_ball_direction: u32, // Whether to randomize ball direction
}

/// Result of match evaluation for round-robin
///
/// # Teaching Note: GPU Result Structure
/// This structure is designed for efficient GPU-CPU data transfer:
/// - Aligned memory layout for optimal transfer speed
/// - Complete match information in a single structure
/// - Minimal data types to reduce bandwidth requirements
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct MatchResult {
    player1_primary: u32,   // Player 1 primary fitness
    player1_secondary: u32, // Player 1 secondary fitness  
    player2_primary: u32,   // Player 2 primary fitness
    player2_secondary: u32, // Player 2 secondary fitness
}

impl GpuContext {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|e| format!("Failed to find an appropriate GPU adapter: {:?}", e))?;
        
        // Get GPU information for optimization
        let adapter_info = adapter.get_info();
        let max_compute_units = adapter.limits().max_compute_workgroups_per_dimension;
        
        info!("GPU Adapter: {} ({})", adapter_info.name, adapter_info.backend);
        info!("Max compute workgroups: {}", max_compute_units);

        let (device, queue) = block_on(adapter.request_device(&Default::default()))
            .map_err(|e| format!("Failed to get GPU device: {}", e))?;

        // Load and compile both shaders
        let single_shader_source = include_str!("../shaders/forward.wgsl");
        let batch_shader_source = include_str!("../shaders/batch_tournament.wgsl");
        
        let single_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Pass Shader"),
            source: wgpu::ShaderSource::Wgsl(single_shader_source.into()),
        });

        let batch_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Batch Tournament Shader"),
            source: wgpu::ShaderSource::Wgsl(batch_shader_source.into()),
        });

        // Create pipeline layout for single processing
        let single_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward Pass Pipeline Layout"),
            bind_group_layouts: &[&device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Single Bind Group Layout"),
                    entries: &[
                        // input buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // weights buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // output buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // config uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            )],
            push_constant_ranges: &[],
        });

        // Create pipeline layout for batch processing
        let batch_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Batch Tournament Pipeline Layout"),
            bind_group_layouts: &[&device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Batch Bind Group Layout"),
                    entries: &[
                        // population weights buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // tournament assignments buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // tournament results buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // batch config uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            )],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Forward Pass Pipeline"),
            layout: Some(&single_pipeline_layout),
            module: &single_shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batch Tournament Pipeline"),
            layout: Some(&batch_pipeline_layout),
            module: &batch_shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(GpuContext {
            device,
            queue,
            pipeline,
            batch_pipeline,
            max_compute_units,
        })
    }

    /// Gets optimal workgroup size for batch processing based on GPU capabilities
    ///
    /// # Teaching Note: GTX 1070 Optimization
    /// The GTX 1070 has 1920 CUDA cores arranged in 15 SMs (Streaming Multiprocessors).
    /// Optimal workgroup sizes should be multiples of warp size (32) and fully utilize
    /// the GPU's compute capability. For populations of 4K-8K, we can achieve excellent
    /// GPU utilization with larger workgroups.
    fn get_optimal_workgroup_size(&self, num_tournaments: u32) -> u32 {
        // Optimal workgroup sizes for different GPU architectures
        // GTX 1070: 15 SMs × 128 cores/SM = 1920 cores
        // Optimal: 64-256 threads per workgroup (2-8 warps)
        let optimal_sizes = [256u32, 128u32, 64u32, 32u32];
        
        // Find the largest workgroup size that provides good GPU utilization
        for &workgroup_size in optimal_sizes.iter() {
            let num_workgroups = (num_tournaments + workgroup_size - 1) / workgroup_size;
            
            // Target 8-64 workgroups for optimal GPU utilization
            // (enough to keep all SMs busy without excessive overhead)
            if num_workgroups >= 8 && num_workgroups <= 128 {
                debug!("Selected workgroup size: {} ({} workgroups for {} tournaments)", 
                       workgroup_size, num_workgroups, num_tournaments);
                return workgroup_size;
            }
        }
        
        // Fallback for small populations
        let fallback_size = 32u32;
        debug!("Using fallback workgroup size: {} for {} tournaments", fallback_size, num_tournaments);
        fallback_size
    }
}

/// An individual whose neural network weights are stored and processed on the GPU.
///
/// # Memory and Performance: A Hybrid Approach
/// This engine uses a hybrid memory model to balance performance and implementation simplicity.
///
/// - **GPU-Side Storage**: The full weight set is stored in a `wgpu::Buffer` on the GPU.
///   This allows the `forward_propagate` method to be extremely fast, as the compute shader
///   can access the weights directly without any data transfer.
///
/// - **CPU-Side Cache**: A `Vec<f32>` of the weights is also kept on the CPU. This is critical
///   for the genetic algorithm. Operations like `crossover` and `mutate` need to read and
///   write individual weights. Reading from the GPU is a slow, synchronous operation that
///   would create a major bottleneck. By operating on the CPU cache, these methods are fast.
///   The `mutate` method then efficiently writes the updated weights back to the GPU in a
///   single bulk transfer.
///
/// # Teaching Note: GPGPU Programming Patterns
/// This design highlights a common pattern in GPGPU programming: minimize data transfer between
/// the host (CPU) and the device (GPU). The ideal is to send data to the GPU once, perform as
/// much computation as possible, and then read the final result back. This engine follows that
/// principle for the forward pass, while the CPU cache provides an escape hatch for the granular
/// manipulations required by the genetic algorithm.
///
/// # Optimization: Buffer Pre-Allocation
/// To maximize performance, each `GpuIndividual` pre-allocates all necessary GPU buffers
/// (`weights`, `input`, `output`, `config`, `staging`) and the `bind_group` upon creation.
/// The `forward_propagate` method then reuses these buffers, only writing new data to the
/// input and config buffers via `queue.write_buffer`. This avoids the significant overhead
/// of creating and destroying buffers on every forward pass.
pub struct GpuIndividual<'a> {
    context: &'a GpuContext,
    weights: Vec<f32>, // CPU-side cache of weights
    weights_buffer: Arc<wgpu::Buffer>,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    config_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Clone for GpuIndividual<'_> {
    /// Creates a deep copy of the individual by cloning the CPU-side weights
    /// and creating a new corresponding GPU buffer.
    ///
    /// # Teaching Note: Clone Semantics for GPU Resources
    /// Cloning GPU resources requires careful consideration. We can't simply copy GPU buffer
    /// handles, as each individual needs its own independent buffers. This implementation
    /// creates entirely new GPU resources for the cloned individual, ensuring complete
    /// independence between the original and clone.
    fn clone(&self) -> Self {
        GpuIndividual::from_weights(&self.weights)
            .expect("Failed to clone GpuIndividual - GPU context unavailable")
    }
}

impl Individual for GpuIndividual<'_> {
    /// Performs a forward pass on the GPU using a compute shader.
    /// This implementation reuses pre-allocated buffers for maximum performance.
    ///
    /// # Teaching Note: GPU Compute Pipeline
    /// The GPU forward pass follows these steps:
    /// 1. **Data Upload**: Input and configuration data are uploaded to GPU buffers
    /// 2. **Compute Dispatch**: A compute shader is launched to process the data in parallel
    /// 3. **Result Copy**: Output is copied to a staging buffer for CPU access
    /// 4. **Synchronization**: CPU waits for GPU to complete and reads the result
    ///
    /// This process is highly optimized for throughput when processing many individuals,
    /// but has overhead that makes it slower than CPU for single forward passes.
    fn forward_propagate(
        &self,
        input: &[f32; INPUT_SIZE],
        activation: Activation,
    ) -> [f32; OUTPUT_SIZE] {
        let queue = &self.context.queue;

        // Update input and config buffers with new data for this pass.
        // CRITICAL: The activation_type mapping MUST match the WGSL shader exactly
        let config_data = GpuConfig {
            activation_type: match activation {
                Activation::ClampedLinear => 0,  // Must match WGSL case 0u
                Activation::Tanh => 1,           // Must match WGSL case 1u  
                Activation::Relu => 2,           // Must match WGSL case 2u
                Activation::Atan => 3,           // Must match WGSL case 3u
                Activation::Linear => 4,         // Must match WGSL case 4u
                Activation::Sigmoid => 5,        // Must match WGSL case 5u
            },
        };
        queue.write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(input));
        queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&config_data));

        // Dispatch the compute shader.
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        } // cpass is dropped, releasing the borrow on encoder

        // Copy the result from the output buffer to the staging buffer.
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.output_buffer.size(),
        );
        queue.submit(Some(encoder.finish()));

        // Read the result from the staging buffer back to the CPU.
        let mut output = [0.0; OUTPUT_SIZE];
        if let Err(e) = read_buffer_sync(&self.context.device, &self.staging_buffer, |buffer_slice| {
            let floats: &[f32] = bytemuck::cast_slice(buffer_slice);
            output.copy_from_slice(&floats[..OUTPUT_SIZE]);
        }) {
            warn!("GPU forward propagation read failed: {}. Returning zero output.", e);
            // Return zero output to allow evolution to continue
            output = [0.0; OUTPUT_SIZE];
        }

        output
    }

    /// # Teaching Note: CPU-Side Crossover for Hybrid Architecture
    /// Crossover is performed entirely on the CPU using the cached `weights` vector. This avoids
    /// a slow round-trip to the GPU. A new child individual is created from the resulting
    /// weight vector, which in turn creates a new buffer on the GPU and copies the data over.
    /// While this creates some GPU overhead, it's much faster than trying to perform crossover
    /// operations directly on GPU buffers.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let mut child_weights = self.weights.clone();
        let parent2_weights = other.weights_as_slice();

        for i in 0..child_weights.len() {
            if rng.random() {
                child_weights[i] = parent2_weights[i];
            }
        }

        GpuIndividual::from_weights(&child_weights)
            .unwrap_or_else(|_| {
                warn!("Failed to create child from crossover, falling back to parent clone");
                self.clone()
            })
    }

    /// # Teaching Note: CPU-Side Mutation and GPU Synchronization
    /// Like crossover, mutation is performed on the fast, CPU-cached `weights` vector. After
    /// the weights are modified, `sync_weights_to_gpu` is called to perform an efficient
    /// bulk transfer of the updated data to the corresponding GPU buffer, ensuring the two
    /// copies remain synchronized.
    ///
    /// # Synchronization Strategy
    /// Rather than synchronizing individual weight changes, we batch all mutations and then
    /// perform a single bulk transfer to the GPU. This minimizes the expensive CPU-GPU
    /// communication while maintaining consistency.
    fn mutate<R: Rng>(&mut self, rng: &mut R, config: &Config) {
        // Mutate the CPU-side cache directly for performance.
        match config.mutation_strategy {
            crate::config::MutationStrategy::CppEquivalent => {
                // Conservative strategy: mutate exactly one randomly selected gene
                let gene_index = rng.random_range(0..self.weights.len());
                let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
                let mutation = normal.sample(rng);
                self.weights[gene_index] += mutation;
            }
            crate::config::MutationStrategy::Modern => {
                // Modern strategy: probabilistic mutation of multiple genes
                for i in 0..self.weights.len() {
                    if rng.random::<f32>() < config.mutation_rate {
                        self.weights[i] += rng.random_range(-1.0..=1.0) * config.mutation_strength;
                    }
                }
            }
        }
        // Sync the entire updated weight vector to the GPU in one go.
        self.sync_weights_to_gpu();
    }

    fn weights_as_slice(&self) -> &[f32] {
        &self.weights
    }

    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        // Note: If this slice is mutated, the caller is responsible for calling `sync_weights_to_gpu`
        // to maintain consistency between the CPU cache and the GPU buffer. The `Individual`
        // trait's default `mutate` implementation does not do this, which is why `GpuIndividual`
        // provides its own `mutate` implementation that correctly handles synchronization.
        &mut self.weights
    }
}

impl Default for GpuIndividual<'_> {
    /// Creates a `GpuIndividual` with randomly initialized weights and prepares GPU buffers.
    ///
    /// # Teaching Note: Graceful Degradation
    /// If GPU initialization fails, this implementation panics. In a production system,
    /// you might want to fall back to a CPU implementation instead. This design choice
    /// makes the failure explicit rather than silently degrading performance.
    fn default() -> Self {
        let mut rng = rand::rng();
        let mut new_weights = Vec::with_capacity(TOTAL_WEIGHTS);
        for _ in 0..TOTAL_WEIGHTS {
            new_weights.push(rng.random_range(-1.0..=1.0));
        }
        GpuIndividual::from_weights(&new_weights)
            .expect("Failed to create default GpuIndividual - GPU context unavailable. Ensure you have a compatible GPU with proper drivers.")
    }
}

impl<'a> GpuIndividual<'a> {
    /// Helper to create a new `GpuIndividual` from a slice of weights.
    /// This involves creating all necessary GPU buffers and copying the weights to the device,
    /// as well as cloning the weights for the CPU-side cache.
    ///
    /// # Error Handling
    /// Returns an error if the GPU context is unavailable or buffer creation fails.
    /// This allows callers to handle GPU failures gracefully rather than panicking.
    fn from_weights(weights: &[f32]) -> Result<Self, Box<dyn std::error::Error>> {
        let context = GPU_CONTEXT.as_ref()
            .ok_or("GPU context is not available. Ensure you have a compatible GPU.")?;
        let device = &context.device;

        // Create all necessary buffers once.
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weights Buffer"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: (INPUT_SIZE * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer_size = (OUTPUT_SIZE * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config Buffer"),
            size: std::mem::size_of::<GpuConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create the bind group once.
        let bind_group_layout = context.pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            context,
            weights: weights.to_vec(),
            weights_buffer: Arc::new(weights_buffer),
            input_buffer,
            output_buffer,
            config_buffer,
            staging_buffer,
            bind_group,
        })
    }

    /// Synchronizes the CPU-side weight cache to the GPU buffer.
    ///
    /// # Teaching Note: Efficient Bulk Transfer
    /// This method performs a single, efficient bulk transfer of all weights from CPU to GPU.
    /// Individual weight updates would be much slower due to the overhead of each GPU operation.
    /// This pattern of batching updates is crucial for GPU performance.
    fn sync_weights_to_gpu(&self) {
        self.context.queue.write_buffer(
            &self.weights_buffer,
            0,
            bytemuck::cast_slice(&self.weights),
        );
    }



}

/// Simplified GPU round-robin engine that exactly matches CPU behavior
///
/// # Teaching Note: Proper GPU-CPU Hybrid Architecture
/// This implementation demonstrates the correct approach:
/// - **GPU accelerates entire round-robin**: Uses existing round_robin.wgsl shader
/// - **Identical game physics**: Matches CPU GameState engine exactly
/// - **Proper fitness functions**: Same formulas as CPU version
/// - **Accurate match counting**: Reports real matches performed
///
/// This ensures identical behavior to CPU version while gaining GPU acceleration.
pub struct GpuBatchEngine {
    context: &'static GpuContext,
    max_population_size: usize,
    // GPU buffers for round-robin processing  
    population_weights_buffer: Option<wgpu::Buffer>,
    match_assignments_buffer: Option<wgpu::Buffer>,
    match_results_buffer: Option<wgpu::Buffer>,
    batch_config_buffer: wgpu::Buffer,
    staging_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
}

impl GpuBatchEngine {
    /// Creates a new round-robin processing engine
    pub fn new(max_population_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let context = GPU_CONTEXT.as_ref()
            .ok_or("GPU context is not available. Ensure you have a compatible GPU.")?;
        
        info!("Initializing GPU round-robin engine for max population size: {}", max_population_size);
        
        // Create config buffer (reused across batches)
        let batch_config_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Config Buffer"),
            size: std::mem::size_of::<BatchConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuBatchEngine {
            context,
            max_population_size,
            population_weights_buffer: None,
            match_assignments_buffer: None,
            match_results_buffer: None,
            batch_config_buffer,
            staging_buffer: None,
            bind_group: None,
        })
    }

    /// Evaluates population using full round-robin tournament exactly like CPU version
    ///
    /// # Teaching Note: Full Round-Robin Implementation
    /// This approach:
    /// 1. Generates all n*(n-1) match pairs like CPU version
    /// 2. Runs each match on GPU with identical game physics
    /// 3. Uses same fitness functions as CPU version
    /// 4. Produces identical results to CPU version
    pub fn evaluate_population<T: Individual>(
        &mut self,
        individuals: &[T],
        config: &Config,
        _tournament_size: usize, // Ignored for round-robin
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let population_size = individuals.len();
        debug!("Starting GPU round-robin evaluation for {} individuals", population_size);
        
        if population_size < 2 {
            return Ok(vec![0.0f32; population_size]);
        }

        // Calculate total matches for full round-robin: n*(n-1)
        let total_matches = population_size * (population_size - 1);
        debug!("Total round-robin matches: {}", total_matches);

        // Prepare GPU resources
        self.prepare_buffers(population_size, total_matches)?;

        // Upload population weights to GPU
        let mut all_weights = Vec::with_capacity(population_size * TOTAL_WEIGHTS);
        for individual in individuals {
            all_weights.extend_from_slice(individual.weights_as_slice());
        }

        self.context.queue.write_buffer(
            self.population_weights_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&all_weights),
        );

        // Generate round-robin match assignments: all pairs (i,j) where i≠j
        let mut match_assignments = Vec::with_capacity(total_matches * 2);
        for i in 0..population_size {
            for j in 0..population_size {
                if i != j {
                    match_assignments.push(i as u32);
                    match_assignments.push(j as u32);
                }
            }
        }

        self.context.queue.write_buffer(
            self.match_assignments_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&match_assignments),
        );

        // Execute round-robin processing on GPU in chunks to avoid TDR
        let optimal_workgroup_size = self.context.get_optimal_workgroup_size(total_matches as u32);
        let total_workgroups = (total_matches as u32 + optimal_workgroup_size - 1) / optimal_workgroup_size;
        
        const MAX_WORKGROUPS_PER_DISPATCH: u32 = 256; // Keep each dispatch reasonably small

        for workgroup_offset in (0..total_workgroups).step_by(MAX_WORKGROUPS_PER_DISPATCH as usize) {
            let workgroups_in_chunk = (total_workgroups - workgroup_offset).min(MAX_WORKGROUPS_PER_DISPATCH);

            // Configure round-robin processing for the current chunk
            let batch_config = BatchConfig {
                population_size: population_size as u32,
                total_matches: total_matches as u32,
                activation_type: match config.activation {
                    Activation::ClampedLinear => 0,
                    Activation::Tanh => 1,
                    Activation::Relu => 2,
                    Activation::Atan => 3,
                    Activation::Linear => 4,
                    Activation::Sigmoid => 5,
                },
                random_seed: rand::rng().random(),
                fitness_function: match config.fitness_func {
                    crate::config::FitnessFunc::CppEquivalent => 0,
                    crate::config::FitnessFunc::ReturnFocused => 1,
                    crate::config::FitnessFunc::VictoryOptimized => 2,
                },
                workgroup_offset,
                random_ball_direction: if config.random_ball_direction { 1 } else { 0 },
            };

            self.context.queue.write_buffer(
                &self.batch_config_buffer,
                0,
                bytemuck::bytes_of(&batch_config),
            );

            let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Round-Robin Chunk Encoder (Offset: {})", workgroup_offset)),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Round-Robin Chunk Pass (Offset: {})", workgroup_offset)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.context.batch_pipeline);
                cpass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
                cpass.dispatch_workgroups(workgroups_in_chunk, 1, 1);
            }
            self.context.queue.submit(Some(encoder.finish()));
        }

        // After all chunks are dispatched, copy results
        let mut final_encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Final Result Copy Encoder"),
        });

        final_encoder.copy_buffer_to_buffer(
            self.match_results_buffer.as_ref().unwrap(),
            0,
            self.staging_buffer.as_ref().unwrap(),
            0,
            (total_matches * std::mem::size_of::<MatchResult>()) as u64,
        );

        self.context.queue.submit(Some(final_encoder.finish()));

        // Read results back to CPU and accumulate fitness exactly like CPU version
        let mut fitness_values = vec![0u64; population_size];
        match read_buffer_sync(&self.context.device, self.staging_buffer.as_ref().unwrap(), |buffer_slice| {
            let results: &[MatchResult] = bytemuck::cast_slice(buffer_slice);
            let mut match_idx = 0;
            
            // Accumulate fitness for each individual from all their matches
            for i in 0..population_size {
                for j in 0..population_size {
                    if i == j { continue; }
                    
                    if match_idx < results.len() {
                        let result = &results[match_idx];
                        
                        // Pack fitness like CPU version: primary << 32 | secondary
                        let i_fitness = ((result.player1_primary as u64) << 32) | (result.player1_secondary as u64);
                        let j_fitness = ((result.player2_primary as u64) << 32) | (result.player2_secondary as u64);
                        
                        fitness_values[i] += i_fitness;
                        fitness_values[j] += j_fitness;
                    }
                    match_idx += 1;
                }
            }
        }) {
            Ok(()) => {
                // Convert packed fitness to f32 for return (extract primary fitness)
                let final_fitness: Vec<f32> = fitness_values.iter()
                    .map(|&packed| (packed >> 32) as f32)
                    .collect();
                
                debug!("GPU round-robin evaluation completed");
                Ok(final_fitness)
            }
            Err(e) => {
                warn!("GPU buffer read failed: {}. Population fitness will be set to 0.", e);
                Ok(vec![0.0f32; population_size])
            }
        }
    }

    /// Prepares GPU buffers for round-robin processing.
    fn prepare_buffers(&mut self, population_size: usize, total_matches: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut reallocate_bind_group = false;

        // Population weights buffer
        let weights_buffer_size = (population_size * TOTAL_WEIGHTS * std::mem::size_of::<f32>()) as u64;
        if self.population_weights_buffer.as_ref().map_or(true, |b| b.size() < weights_buffer_size) {
            info!("Allocating population weights buffer: {} bytes", weights_buffer_size);
            let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Population Weights Buffer"),
                size: weights_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.population_weights_buffer = Some(buffer);
            reallocate_bind_group = true;
        }
        
        // Match assignments buffer (pairs of individual indices)
        let assignments_buffer_size = (total_matches * 2 * std::mem::size_of::<u32>()) as u64;
        if self.match_assignments_buffer.as_ref().map_or(true, |b| b.size() < assignments_buffer_size) {
            info!("Allocating match assignments buffer: {} bytes", assignments_buffer_size);
            let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Match Assignments Buffer"),
                size: assignments_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.match_assignments_buffer = Some(buffer);
            reallocate_bind_group = true;
        }

        // Match results buffer
        let results_buffer_size = (total_matches * std::mem::size_of::<MatchResult>()) as u64;
        if self.match_results_buffer.as_ref().map_or(true, |b| b.size() < results_buffer_size) {
            info!("Allocating match results buffer: {} bytes", results_buffer_size);
            let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Match Results Buffer"),
                size: results_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.match_results_buffer = Some(buffer);
            reallocate_bind_group = true;
        }

        // Staging buffer for reading results
        if self.staging_buffer.as_ref().map_or(true, |b| b.size() < results_buffer_size) {
            info!("Allocating staging buffer: {} bytes", results_buffer_size);
            let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: results_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.staging_buffer = Some(buffer);
        }

        // Re-create bind group if any buffers were reallocated
        if self.bind_group.is_none() || reallocate_bind_group {
            info!("Creating GPU bind group for round-robin processing.");
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Round-Robin Processing Bind Group"),
                layout: &self.context.batch_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.population_weights_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.match_assignments_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.match_results_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.batch_config_buffer.as_entire_binding(),
                    },
                ],
            });
            self.bind_group = Some(bind_group);
        }
        
        Ok(())
    }
}

/// Helper function to synchronously read data from a wgpu buffer.
///
/// # Teaching Note: Asynchronous GPU Operations
/// GPU operations are inherently asynchronous - the CPU submits commands and the GPU
/// executes them later. This function bridges that gap by using a channel to wait
/// for the GPU operation to complete before returning the data to the caller.
/// This pattern is necessary when you need the results immediately (like for debugging)
/// but should be avoided in performance-critical code.
fn read_buffer_sync(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    mut callback: impl FnMut(&[u8]),
) -> Result<(), Box<dyn std::error::Error>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = buffer.slice(..);

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        if let Err(e) = &result {
            error!("Failed to map GPU buffer for reading: {:?}", e);
        }
        let _ = sender.send(result);
    });

    // Optimized polling with minimal CPU overhead
    let mut poll_count = 0;
    loop {
        match receiver.try_recv() {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                callback(&data);
                drop(data);
                buffer.unmap();
                return Ok(());
            }
            Ok(Err(e)) => {
                error!("GPU buffer mapping failed: {:?}", e);
                buffer.unmap(); // Ensure buffer is unmapped on error
                return Err(format!("Failed to read data from GPU buffer: {:?}", e).into());
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Adaptive polling strategy to reduce CPU overhead
                poll_count += 1;
                if poll_count < 2 {
                    // Fast polling for the first few iterations
                    let _ = device.poll(wgpu::MaintainBase::Poll);
                } else {
                    // Switch to blocking wait to avoid CPU spinning
                    let _ = device.poll(wgpu::MaintainBase::Wait);
                }
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                error!("Failed to receive GPU operation result: channel disconnected");
                return Err("Failed to read data from GPU buffer due to communication error".into());
            }
        }
    }
}
