//! A neural network engine that leverages the GPU for massively parallel computation.

use crate::config::{Activation, Config};
use crate::{constants::*, traits::Individual};

use bytemuck::{Pod, Zeroable};
use once_cell::sync::Lazy;
use pollster::block_on;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::{
    fs,
    io::{self, Write},
    sync::Arc,
};
use wgpu::util::DeviceExt;

// A global, lazily-initialized GPU context, shared across all GpuIndividuals.
static GPU_CONTEXT: Lazy<GpuContext> = Lazy::new(GpuContext::new);

/// Holds the WGPU device, queue, and the pre-compiled compute pipeline.
///
/// # Teaching Note
/// This struct encapsulates the boilerplate `wgpu` setup. By creating it once with `Lazy`,
/// we avoid the high cost of initializing the GPU for every single `GpuIndividual`.
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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = block_on(adapter.request_device(&Default::default(), None))
            .expect("Failed to get device");

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
            entry_point: "main",
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
/// # Memory and Performance
/// - Weights are held in a `wgpu::Buffer` on the GPU for fast access by compute shaders.
/// - The `forward_propagate` method is extremely fast as it dispatches a shader and doesn't
///   block or transfer data back to the CPU.
/// - Genetic operations (`crossover`, `mutate`) are currently a performance bottleneck.
///   They transfer weights back to the CPU, perform the logic, and transfer them back.
///   This is a candidate for a future optimization using a dedicated compute shader.
pub struct GpuIndividual {
    weights_buffer: Arc<wgpu::Buffer>,
    context: &'static GpuContext,
}

impl Clone for GpuIndividual {
    /// Creates a deep copy of the individual by copying the GPU buffer.
    fn clone(&self) -> Self {
        let new_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloned Weights Buffer"),
            size: self.weights_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Clone Encoder"),
                });
        encoder.copy_buffer_to_buffer(
            &self.weights_buffer,
            0,
            &new_buffer,
            0,
            self.weights_buffer.size(),
        );
        self.context.queue.submit(Some(encoder.finish()));

        Self {
            weights_buffer: Arc::new(new_buffer),
            context: self.context,
        }
    }
}

impl Individual for GpuIndividual {
    fn name() -> &'static str {
        "GPU"
    }

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

    /// # Performance Note
    /// This is a major performance bottleneck. Crossover is performed on the CPU, requiring
    /// a full data transfer from two GPU individuals and a subsequent transfer back to the new child.
    /// This should be implemented in a dedicated compute shader.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let p1_weights = self.get_weights_from_gpu();
        let p2_weights = other.get_weights_from_gpu();
        let mut child_weights = vec![0.0; TOTAL_WEIGHTS];

        for i in 0..TOTAL_WEIGHTS {
            child_weights[i] = if rng.gen() {
                p1_weights[i]
            } else {
                p2_weights[i]
            };
        }

        GpuIndividual::from_weights(&child_weights)
    }

    /// # Performance Note
    /// Like crossover, this is a performance bottleneck due to the GPU-CPU data transfer.
    /// This should be implemented in a compute shader for efficiency.
    fn mutate<R: Rng>(&mut self, rng: &mut R, config: &Config) {
        let mut weights = self.get_weights_from_gpu();
        let normal = Normal::new(0.0, config.mutation_strength).unwrap();

        for i in 0..TOTAL_WEIGHTS {
            if rng.gen::<f32>() < config.mutation_rate {
                weights[i] += normal.sample(rng);
            }
        }
        self.set_weights_on_gpu(&weights);
    }

    fn weights_as_slice(&self) -> &[f32] {
        panic!("Getting a direct slice from GPU memory is not feasible. Use `save` or `get_weights_from_gpu`.");
    }

    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        panic!("Getting a direct mutable slice from GPU memory is not feasible.");
    }

    fn save(&self, path: &str) -> io::Result<()> {
        let weights = self.get_weights_from_gpu();
        let mut file = fs::File::create(path)?;
        let weights_bytes: &[u8] = bytemuck::cast_slice(&weights);
        file.write_all(weights_bytes)
    }
}

impl Default for GpuIndividual {
    fn default() -> Self {
        let mut weights = vec![0.0; TOTAL_WEIGHTS];
        let mut rng = rand::thread_rng();
        for weight in weights.iter_mut() {
            *weight = rng.gen_range(-1.0..=1.0);
        }
        GpuIndividual::from_weights(&weights)
    }
}

impl GpuIndividual {
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
        }
    }

    fn set_weights_on_gpu(&mut self, weights: &[f32]) {
        self.context
            .queue
            .write_buffer(&self.weights_buffer, 0, bytemuck::cast_slice(weights));
    }

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
    GPU_CONTEXT.device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range();
        callback(&data);
        drop(data);
        buffer.unmap();
    } else {
        panic!("Failed to read data from GPU buffer");
    }
}
