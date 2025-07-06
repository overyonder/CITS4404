/// This WGSL shader performs a forward pass for a 3-layer neural network.
/// It's designed to be generic and is controlled by uniforms passed from the host.

// Bindings for data buffers
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Uniforms for configuration
struct Config {
    // 0: Tanh, 1: ReLU, 2: Atan, 3: Linear, 4: Sigmoid
    activation_type: u32,
};
@group(0) @binding(3) var<uniform> config: Config;

// Network architecture constants (must match constants.rs)
const INPUT_SIZE: u32 = 8u;
const L1_SIZE: u32 = 16u;
const L2_SIZE: u32 = 4u;
const OUTPUT_SIZE: u32 = 1u;

const INPUT_SIZE_WITH_BIAS: u32 = INPUT_SIZE + 1u;
const L1_SIZE_WITH_BIAS: u32 = L1_SIZE + 1u;

const L1_WEIGHTS: u32 = L1_SIZE * INPUT_SIZE_WITH_BIAS;
const L2_WEIGHTS: u32 = L2_SIZE * L1_SIZE_WITH_BIAS;

/// Applies the selected activation function.
fn apply_activation(val: f32) -> f32 {
    switch config.activation_type {
        // Tanh
        case 0u: {
            return tanh(val);
        }
        // ReLU
        case 1u: {
            return max(val, 0.0);
        }
        // Atan
        case 2u: {
            return atan(val);
        }
        // Linear
        case 3u: {
            return val;
        }
        // Sigmoid
        case 4u: {
            return 1.0 / (1.0 + exp(-val));
        }
        // Default to linear
        default: {
            return val;
        }
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var l1_outputs: array<f32, L1_SIZE>;
    var l2_outputs: array<f32, L2_SIZE>;

    // Layer 1: Input -> Hidden 1
    for (var i = 0u; i < L1_SIZE; i = i + 1u) {
        let weight_start_index = i * INPUT_SIZE_WITH_BIAS;
        var sum = 0.0;
        for (var j = 0u; j < INPUT_SIZE; j = j + 1u) {
            sum = sum + input[j] * weights[weight_start_index + j];
        }
        sum = sum + weights[weight_start_index + INPUT_SIZE]; // Bias
        l1_outputs[i] = apply_activation(sum);
    }

    // Layer 2: Hidden 1 -> Hidden 2
    for (var i = 0u; i < L2_SIZE; i = i + 1u) {
        let weight_start_index = L1_WEIGHTS + i * L1_SIZE_WITH_BIAS;
        var sum = 0.0;
        for (var j = 0u; j < L1_SIZE; j = j + 1u) {
            sum = sum + l1_outputs[j] * weights[weight_start_index + j];
        }
        sum = sum + weights[weight_start_index + L1_SIZE]; // Bias
        l2_outputs[i] = apply_activation(sum);
    }

    // Layer 3: Hidden 2 -> Output (Linear activation)
    let weight_start_index = L1_WEIGHTS + L2_WEIGHTS;
    var sum = 0.0;
    for (var j = 0u; j < L2_SIZE; j = j + 1u) {
        sum = sum + l2_outputs[j] * weights[weight_start_index + j];
    }
    sum = sum + weights[weight_start_index + L2_SIZE]; // Bias
    output[0] = sum;
}
