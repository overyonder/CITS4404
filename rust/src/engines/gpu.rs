//! An experimental neural network engine that leverages the GPU for massively parallel
//! tournament evaluation using WGSL shaders.
//!
//! # WARNING: Experimental
//! This module is highly experimental and may not be fully functional or stable.
//! It requires a compatible GPU and a modern graphics driver (Vulkan, DX12, Metal).
//!
//! # Teaching Note: GPU Computing in Machine Learning
//! GPUs excel at parallel computation, making them ideal for neural network operations.
//! While this implementation focuses on forward propagation, modern ML frameworks
//! like TensorFlow and PyTorch use similar principles for both forward and backward
//! passes, training massive networks with millions of parameters in parallel.

use crate::{config::Activation, constants::*, traits::Individual, Config};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::Lazy;
use pollster::block_on;
use rand::Rng;
use rand_distr::Distribution;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use tracing::{error, warn};

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

/// Holds the WGPU device, queue, and the pre-compiled compute pipeline.
///
/// # Teaching Note: Global GPU Context and Resource Management
/// GPU resources are expensive to create and should be reused whenever possible.
/// This struct encapsulates the boilerplate `wgpu` setup. Initializing a GPU device and
/// compiling shader pipelines are expensive operations. By creating a single, global context
/// using `once_cell::sync::Lazy`, we ensure this setup cost is paid only once when the first
/// `GpuIndividual` is created. All subsequent individuals will share this context, making their
/// creation nearly instantaneous.
///
/// # Error Handling
/// GPU initialization can fail for various reasons (no compatible GPU, driver issues, etc.).
/// The context creation is now wrapped in a Result to handle these cases gracefully.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
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

impl GpuContext {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|e| format!("Failed to find an appropriate GPU adapter: {}", e))?;

        let (device, queue) = block_on(adapter.request_device(&Default::default()))
            .map_err(|e| format!("Failed to get GPU device: {}", e))?;

        // Load and compile the WGSL shader
        let shader_source = include_str!("../shaders/forward.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Pass Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward Pass Pipeline Layout"),
            bind_group_layouts: &[&device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bind Group Layout"),
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Forward Pass Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(GpuContext {
            device,
            queue,
            pipeline,
        })
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
        read_buffer_sync(&self.staging_buffer, |buffer_slice| {
            let floats: &[f32] = bytemuck::cast_slice(buffer_slice);
            output.copy_from_slice(&floats[..OUTPUT_SIZE]);
        });

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

    /// Reads the weights from the GPU buffer back to the CPU.
    ///
    /// # Teaching Note: GPU-to-CPU Data Transfer
    /// This function demonstrates how to read data back from the GPU. It involves creating a
    /// special "staging" buffer that is accessible by both the CPU and GPU. A command is
    /// encoded to copy from the main weights buffer to the staging buffer. The CPU then has
    /// to wait for the GPU to finish this operation before it can map the staging buffer's
    /// memory and read the data. This is a synchronous and potentially slow operation.
    ///
    /// # Performance Warning
    /// This method is primarily for debugging and should be avoided in performance-critical
    /// code paths. The CPU-side cache should be used for weight access instead.
    #[allow(dead_code)]
    fn get_weights_from_gpu(&self) -> Vec<f32> {
        let staging_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weights Staging Buffer"),
            size: self.weights_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.weights_buffer,
            0,
            &staging_buffer,
            0,
            self.weights_buffer.size(),
        );
        self.context.queue.submit(Some(encoder.finish()));

        let mut result = Vec::new();
        read_buffer_sync(&staging_buffer, |buffer_slice| {
            result = bytemuck::cast_slice::<u8, f32>(buffer_slice).to_vec();
        });
        result
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
fn read_buffer_sync(buffer: &wgpu::Buffer, mut callback: impl FnMut(&[u8])) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = buffer.slice(..);

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        if let Err(e) = &result {
            error!("Failed to map GPU buffer for reading: {:?}", e);
        }
        let _ = sender.send(result);
    });

    match receiver.recv() {
        Ok(Ok(())) => {
            let data = buffer_slice.get_mapped_range();
            callback(&data);
            drop(data);
            buffer.unmap();
        }
        Ok(Err(e)) => {
            error!("GPU buffer mapping failed: {:?}", e);
            panic!("Failed to read data from GPU buffer: {:?}", e);
        }
        Err(e) => {
            error!("Failed to receive GPU operation result: {}", e);
            panic!("Failed to read data from GPU buffer due to communication error: {}", e);
        }
    }
}
