/// GPU shader for neural network forward propagation using WGSL (WebGPU Shading Language).
///
/// # Teaching Note: GPU Parallel Computing
/// This shader demonstrates how neural network computation can be parallelized on the GPU.
/// While this implementation processes one network at a time, the same principles scale
/// to processing thousands of networks simultaneously, which is how modern deep learning
/// frameworks achieve their performance.
///
/// # Architecture
/// This shader implements a 3-layer feedforward neural network:
/// - Input Layer: 8 neurons (game state: paddle positions, ball position/velocity)
/// - Hidden Layer 1: 16 neurons with bias
/// - Hidden Layer 2: 4 neurons with bias  
/// - Output Layer: 1 neuron (paddle movement decision)
///
/// # Critical: Constants Synchronization
/// The constants below MUST match those in constants.rs exactly, or the network
/// will malfunction. In a production system, consider generating this file from
/// the Rust constants to ensure they stay in sync.

// Bindings for data buffers passed from the CPU
@group(0) @binding(0) var<storage, read> input: array<f32>;        // Network inputs
@group(0) @binding(1) var<storage, read> weights: array<f32>;      // All network weights
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // Network outputs

// Uniforms for configuration (changes between shader invocations)
struct Config {
    /// Activation function selector - MUST match gpu.rs GpuConfig exactly:
    /// 0: ClampedLinear, 1: Tanh, 2: ReLU, 3: Atan, 4: Linear, 5: Sigmoid
    activation_type: u32,
};
@group(0) @binding(3) var<uniform> config: Config;

// Network architecture constants - CRITICAL: Must match constants.rs exactly
const INPUT_SIZE: u32 = 8u;    // Number of inputs to the network
const L1_SIZE: u32 = 16u;      // Number of neurons in first hidden layer
const L2_SIZE: u32 = 4u;       // Number of neurons in second hidden layer  
const OUTPUT_SIZE: u32 = 1u;   // Number of network outputs

// Derived constants for weight array indexing
const INPUT_SIZE_WITH_BIAS: u32 = INPUT_SIZE + 1u;  // Input layer + bias term
const L1_SIZE_WITH_BIAS: u32 = L1_SIZE + 1u;        // Hidden layer 1 + bias term

// Weight block sizes (number of weights for each layer)
const L1_WEIGHTS: u32 = L1_SIZE * INPUT_SIZE_WITH_BIAS;  // Layer 1 weight count
const L2_WEIGHTS: u32 = L2_SIZE * L1_SIZE_WITH_BIAS;     // Layer 2 weight count

/// Applies the selected activation function to a scalar value.
///
/// # Teaching Note: Activation Functions on GPU
/// GPU shaders excel at applying the same operation to many values in parallel.
/// Each activation function here could be applied to thousands of neurons
/// simultaneously in a larger network. The switch statement compiles to 
/// efficient conditional execution on the GPU.
///
/// # Performance: Branch Divergence
/// GPU cores execute in groups (warps/wavefronts) where all cores in a group
/// must execute the same instruction. Different activation functions can cause
/// "branch divergence" where some cores idle while others work. In practice,
/// all neurons in a layer typically use the same activation function, avoiding this issue.
fn apply_activation(val: f32) -> f32 {
    switch config.activation_type {
        // ClampedLinear: Linear response with bounds to prevent extreme values
        case 0u: {
            return clamp(val, -1.0, 1.0);
        }
        // Tanh: Smooth S-curve, zero-centered output, good gradient properties
        case 1u: {
            return tanh(val);
        }
        // ReLU: Rectified Linear Unit, simple and fast, prevents vanishing gradients
        case 2u: {
            return max(val, 0.0);
        }
        // Atan: Arctangent, smooth and bounded, alternative to tanh
        case 3u: {
            return atan(val);
        }
        // Linear: No transformation, typically used in output layers
        case 4u: {
            return val;
        }
        // Sigmoid: Classic S-curve, outputs in (0,1), historically important
        case 5u: {
            return 1.0 / (1.0 + exp(-val));
        }
        // Fallback to linear for unknown activation types
        default: {
            return val;
        }
    }
}

/// Main compute shader entry point.
///
/// # Teaching Note: GPU Compute Shaders
/// Compute shaders are programs that run on the GPU for general-purpose computation
/// (GPGPU). Unlike graphics shaders that render pixels, compute shaders can perform
/// arbitrary calculations. This makes them ideal for machine learning workloads.
///
/// # Workgroup Size
/// @workgroup_size(1) means this shader processes one network at a time. In a more
/// advanced implementation, you could process multiple networks in parallel by
/// increasing the workgroup size and using the global_invocation_id to determine
/// which network each thread should process.
///
/// # Memory Layout
/// The weights array contains all network weights in a specific order:
/// [Layer1_Neuron0_Weights..., Layer1_Neuron0_Bias, Layer1_Neuron1_Weights..., ...]
/// This flat layout is cache-friendly and allows for efficient sequential access.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Temporary storage for layer outputs
    // In larger networks, these might be stored in shared memory for efficiency
    var l1_outputs: array<f32, L1_SIZE>;
    var l2_outputs: array<f32, L2_SIZE>;

    // Layer 1: Input -> Hidden 1
    // Each neuron in this layer connects to all inputs plus a bias term
    for (var i = 0u; i < L1_SIZE; i = i + 1u) {
        let weight_start_index = i * INPUT_SIZE_WITH_BIAS;
        var sum = 0.0;
        
        // Compute weighted sum of inputs (dot product)
        for (var j = 0u; j < INPUT_SIZE; j = j + 1u) {
            sum = sum + input[j] * weights[weight_start_index + j];
        }
        
        // Add bias term (stored after the input weights for this neuron)
        sum = sum + weights[weight_start_index + INPUT_SIZE];
        
        // Apply activation function and store result
        l1_outputs[i] = apply_activation(sum);
    }

    // Layer 2: Hidden 1 -> Hidden 2  
    // Each neuron connects to all Layer 1 outputs plus a bias term
    for (var i = 0u; i < L2_SIZE; i = i + 1u) {
        let weight_start_index = L1_WEIGHTS + i * L1_SIZE_WITH_BIAS;
        var sum = 0.0;
        
        // Compute weighted sum of Layer 1 outputs
        for (var j = 0u; j < L1_SIZE; j = j + 1u) {
            sum = sum + l1_outputs[j] * weights[weight_start_index + j];
        }
        
        // Add bias term
        sum = sum + weights[weight_start_index + L1_SIZE];
        
        // Apply activation function and store result
        l2_outputs[i] = apply_activation(sum);
    }

    // Layer 3: Hidden 2 -> Output (typically linear for regression tasks)
    // Single output neuron connects to all Layer 2 outputs plus bias
    let weight_start_index = L1_WEIGHTS + L2_WEIGHTS;
    var sum = 0.0;
    
    // Compute weighted sum of Layer 2 outputs
    for (var j = 0u; j < L2_SIZE; j = j + 1u) {
        sum = sum + l2_outputs[j] * weights[weight_start_index + j];
    }
    
    // Add bias term (final weight in the array)
    sum = sum + weights[weight_start_index + L2_SIZE];
    
    // Output layer typically uses linear activation for continuous control
    // The value represents paddle movement direction and intensity
    output[0] = sum;
}
