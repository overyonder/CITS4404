//! A neural network engine that leverages the GPU for massively parallel computation.

use crate::config::{Activation, Config};
use crate::{constants::*, traits::Individual};

use bytemuck::{Pod, Zeroable};
use once_cell::sync::Lazy;
use pollster::block_on;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::{fs, io::Read, sync::Arc};
use wgpu::util::DeviceExt;

// A global, lazily-initialized GPU context, shared across all GpuIndividuals.
static GPU_CONTEXT: Lazy<GpuContext> = Lazy::new(GpuContext::new);

/// Holds the WGPU device, queue, and the pre-compiled compute pipeline.
///
/// # Teaching Note: Global GPU Context
/// This struct encapsulates the boilerplate `wgpu` setup. Initializing a GPU device and
/// compiling shader pipelines are expensive operations. By creating a single, global context
/// using `once_cell::sync::Lazy`, we ensure this setup cost is paid only once when the first
/// `GpuIndividual` is created. All subsequent individuals will share this context, making their
/// creation nearly instantaneous.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

/// Uniform data passed from the CPU to the GPU shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuConfig {
    /// 0: Tanh, 1: ReLU, 2: Atan, 3: Linear
    activation_type: u32,
}

impl GpuContext {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to find an appropriate adapter");

        let (device, queue) =
            block_on(adapter.request_device(&Default::default())).expect("Failed to get device");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Pass Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/forward.wgsl").into()),
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

        GpuContext {
            device,
            queue,
            pipeline,
        }
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
/// # Teaching Note
/// This design highlights a common pattern in GPGPU programming: minimize data transfer between
/// the host (CPU) and the device (GPU). The ideal is to send data to the GPU once, perform as
/// much computation as possible, and then read the final result back. This engine follows that
/// principle for the forward pass, while the CPU cache provides an escape hatch for the granular
/// manipulations required by the genetic algorithm.
pub struct GpuIndividual<'a> {
    weights_buffer: Arc<wgpu::Buffer>,
    context: &'a GpuContext,
    weights: Vec<f32>, // CPU-side cache of weights
}

impl Clone for GpuIndividual<'_> {
    /// Creates a deep copy of the individual by cloning the CPU-side weights
    /// and creating a new corresponding GPU buffer.
    fn clone(&self) -> Self {
        GpuIndividual::from_weights(&self.weights)
    }
}

impl Individual for GpuIndividual<'_> {
    /// Performs a forward pass on the GPU using a compute shader.
    fn forward_propagate(
        &self,
        input: &[f32; INPUT_SIZE],
        activation: Activation,
    ) -> [f32; OUTPUT_SIZE] {
        let device = &self.context.device;
        let queue = &self.context.queue;

        // Create buffers for input, output, and config
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer_size = (OUTPUT_SIZE * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let config_data = GpuConfig {
            activation_type: match activation {
                Activation::Tanh => 0,
                Activation::Relu => 1,
                Activation::Atan => 2,
                Activation::Linear => 3,
            },
        };
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&config_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create a bind group to link buffers to shader bindings
        let bind_group_layout = self.context.pipeline.get_bind_group_layout(0);
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
                    resource: self.weights_buffer.as_entire_binding(),
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

        // Dispatch the compute shader
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.context.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
        drop(cpass);

        // Create a staging buffer to read the output back to the CPU
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        queue.submit(Some(encoder.finish()));

        // Read the result from the staging buffer
        let mut output = [0.0; OUTPUT_SIZE];
        read_buffer_sync(&staging_buffer, |buffer_slice| {
            let floats: &[f32] = bytemuck::cast_slice(buffer_slice);
            output.copy_from_slice(&floats[..OUTPUT_SIZE]);
        });

        output
    }

    /// # Teaching Note: CPU-Side Crossover
    /// Crossover is performed entirely on the CPU using the cached `weights` vector. This avoids
    /// a slow round-trip to the GPU. A new child individual is created from the resulting
    /// weight vector, which in turn creates a new buffer on the GPU and copies the data over.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let p1_weights = &self.weights;
        let p2_weights = &other.weights;
        let mut child_weights = vec![0.0; TOTAL_WEIGHTS];

        for i in 0..TOTAL_WEIGHTS {
            child_weights[i] = if rng.random::<bool>() {
                p1_weights[i]
            } else {
                p2_weights[i]
            };
        }

        GpuIndividual::from_weights(&child_weights)
    }

    /// # Teaching Note: CPU-Side Mutation and GPU Synchronization
    /// Like crossover, mutation is performed on the fast, CPU-cached `weights` vector. After
    /// the weights are modified, `set_weights_on_gpu` is called to perform an efficient
    /// bulk transfer of the updated data to the corresponding GPU buffer, ensuring the two
    /// copies remain synchronized.
    fn mutate<R: Rng>(&mut self, rng: &mut R, config: &Config) {
        let normal = Normal::new(0.0, config.mutation_strength as f64).unwrap();

        for i in 0..TOTAL_WEIGHTS {
            if rng.random::<f32>() < config.mutation_rate {
                self.weights[i] += normal.sample(rng) as f32;
            }
        }
        // After mutating the CPU-side cache, update the GPU buffer.
        // A clone is created here to work around a borrow checker limitation,
        // avoiding a simultaneous mutable and immutable borrow of `self`.
        self.set_weights_on_gpu(&self.weights.clone());
    }

    fn weights_as_slice(&self) -> &[f32] {
        &self.weights
    }

    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        // Note: If this slice is mutated, the caller is responsible for calling `set_weights_on_gpu`
        // to maintain consistency between the CPU cache and the GPU buffer. The `Individual`
        // trait's default `mutate` implementation does not do this, which is why `GpuIndividual`
        // provides its own `mutate` implementation that correctly handles synchronization.
        &mut self.weights
    }

    /// Loads a `GpuIndividual` and its configuration from a file.
    ///
    /// # File Format
    /// The function expects the file to be in the format created by the `save` method:
    /// 1. `u64` (little-endian): Length of the JSON config string.
    /// 2. `[u8]`: The UTF-8 encoded JSON config string.
    /// 3. `[f32]`: The raw `f32` weights.
    ///
    /// # Implementation
    /// The weights are first loaded into a heap-allocated `Vec<f32>` on the CPU.
    /// Then, the `from_weights` constructor is used to create the `GpuIndividual`,
    /// which handles creating a `wgpu::Buffer` and copying the weights to the GPU.
    ///
    /// # Returns
    /// A `Result` containing a tuple of the loaded `GpuIndividual` and its `Config`,
    /// or an error if reading or deserialization fails.
    ///
    /// # Teaching Note
    /// This function demonstrates how to load data for a GPU-based resource. The raw bytes are
    /// first loaded into a standard `Vec<f32>` on the CPU. Then, the `from_weights` constructor
    // is called, which encapsulates the logic of creating a new GPU buffer and queueing a
    // command to copy the host-side data to the device.
    fn load(path: &str) -> Result<(Self, Config), Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let mut file = fs::File::open(path)?;

        // 1. Read config length
        let mut config_len_bytes = [0u8; 8];
        file.read_exact(&mut config_len_bytes)?;
        let config_len = u64::from_le_bytes(config_len_bytes);

        // 2. Read and deserialize config
        let mut config_bytes = vec![0u8; config_len as usize];
        file.read_exact(&mut config_bytes)?;
        let config: Config = serde_json::from_slice(&config_bytes)?;

        // 3. Read weights
        let mut weights_bytes = Vec::new();
        file.read_to_end(&mut weights_bytes)?;

        if weights_bytes.len() != TOTAL_WEIGHTS * std::mem::size_of::<f32>() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Expected {} weight bytes, but found {}",
                    TOTAL_WEIGHTS * std::mem::size_of::<f32>(),
                    weights_bytes.len()
                ),
            )));
        }

        let weights: Vec<f32> = bytemuck::cast_slice(&weights_bytes).to_vec();

        // Use the from_weights constructor to correctly initialize the GPU buffer
        let individual = GpuIndividual::from_weights(&weights);

        Ok((individual, config))
    }
}

impl Default for GpuIndividual<'_> {
    /// Creates a `GpuIndividual` with random weights, initializing them on both the CPU and GPU.
    ///
    /// # Teaching Note
    /// The `Default` trait is used by `Population::new` to create the initial population.
    /// This implementation first creates a random set of weights on the CPU (heap), and then
    /// uses `from_weights` to create the individual, which also copies the initial weights
    /// to a new buffer on the GPU.
    fn default() -> Self {
        let mut weights = vec![0.0; TOTAL_WEIGHTS];
        let mut rng = rand::rng();
        for weight in weights.iter_mut() {
            *weight = rng.random_range(-1.0..=1.0);
        }
        GpuIndividual::from_weights(&weights)
    }
}

impl<'a> GpuIndividual<'a> {
    /// Creates a new `GpuIndividual` from a slice of weights.
    /// This involves creating a GPU buffer and copying the weights to it,
    /// as well as cloning the weights for the CPU-side cache.
    fn from_weights(weights: &[f32]) -> Self {
        let context = &*GPU_CONTEXT;
        let weights_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Weights Buffer"),
                contents: bytemuck::cast_slice(weights),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        Self {
            weights_buffer: Arc::new(weights_buffer),
            context,
            weights: weights.to_vec(),
        }
    }

    /// Writes the provided weight slice to the GPU buffer.
    fn set_weights_on_gpu(&mut self, weights: &[f32]) {
        self.context
            .queue
            .write_buffer(&self.weights_buffer, 0, bytemuck::cast_slice(weights));
    }

    /// Reads the weights from the GPU buffer back to the CPU.
    ///
    /// # Teaching Note
    /// This function demonstrates how to read data back from the GPU. It involves creating a
    /// special "staging" buffer that is accessible by both the CPU and GPU. A command is
    /// encoded to copy from the main weights buffer to the staging buffer. The CPU then has
    /// to wait for the GPU to finish this operation before it can map the staging buffer's
    /// memory and read the data. This is a synchronous and potentially slow operation.
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
fn read_buffer_sync(buffer: &wgpu::Buffer, mut callback: impl FnMut(&[u8])) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = buffer.slice(..);

    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        sender.send(v).unwrap();
    });

    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range();
        callback(&data);
        drop(data);
        buffer.unmap();
    } else {
        panic!("Failed to read data from GPU buffer");
    }
}
