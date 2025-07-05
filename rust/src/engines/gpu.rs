use crate::config::EvolutionConfig;
use crate::{constants::*, traits::Individual};

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

// A global, lazily-initialized GPU context.
static GPU_CONTEXT: Lazy<GpuContext> = Lazy::new(GpuContext::new);

/// Holds the WGPU device, queue, and the pre-compiled compute pipeline.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl GpuContext {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find an appropriate adapter");

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .expect("Failed to get device");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Pass Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/forward.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Forward Pass Pipeline"),
            layout: None, // Inferred from shader
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

/// An individual whose weights are stored and processed on the GPU.
pub struct GpuIndividual {
    weights_buffer: Arc<wgpu::Buffer>,
    context: &'static GpuContext,
}

impl Clone for GpuIndividual {
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

    fn forward(&self, input: &[f32; INPUT_SIZE], _config: &EvolutionConfig) -> [f32; OUTPUT_SIZE] {
        let device = &self.context.device;
        let queue = &self.context.queue;

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (OUTPUT_SIZE * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &self.context.pipeline.get_bind_group_layout(0),
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
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);

        let mut output = [0.0; OUTPUT_SIZE];
        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            output.copy_from_slice(&floats[..OUTPUT_SIZE]);
            drop(data);
            staging_buffer.unmap();
        } else {
            panic!("Failed to read GPU output");
        }

        output
    }

    fn recombine_from<R: Rng>(
        &mut self,
        p1: &Self,
        p2: &Self,
        rng: &mut R,
        config: &EvolutionConfig,
    ) {
        // FIXME: This is a major performance bottleneck. The recombination logic should
        // be implemented in a compute shader to avoid transferring data between the CPU
        // and GPU.

        let p1_weights = p1.get_weights_from_gpu();
        let p2_weights = p2.get_weights_from_gpu();
        let mut new_weights = vec![0.0; TOTAL_WEIGHTS];

        let normal = Normal::new(0.0, config.mutation_strength).unwrap();
        for i in 0..TOTAL_WEIGHTS {
            new_weights[i] = if rng.gen() {
                p1_weights[i]
            } else {
                p2_weights[i]
            };
            if rng.gen::<f32>() < config.mutation_rate {
                new_weights[i] += normal.sample(rng);
            }
        }
        self.set_weights_on_gpu(&new_weights);
    }

    fn weights_as_slice(&self) -> &[f32] {
        unimplemented!(
            "Getting a direct slice from GPU memory is not efficient. Override save/load."
        );
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
        let device = &self.context.device;
        let queue = &self.context.queue;
        let buffer = &self.weights_buffer;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weights Staging Buffer"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to read weights from GPU");
        }
    }
}
