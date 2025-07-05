@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const L1_SIZE: u32 = 16u;
const L2_SIZE: u32 = 4u;

const INPUT_SIZE_WITH_BIAS: u32 = 8u + 1u;
const L1_SIZE_WITH_BIAS: u32 = L1_SIZE + 1u;
const L2_SIZE_WITH_BIAS: u32 = L2_SIZE + 1u;

const L1_WEIGHTS: u32 = L1_SIZE * INPUT_SIZE_WITH_BIAS;
const L2_WEIGHTS: u32 = L2_SIZE * L1_SIZE_WITH_BIAS;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var l1_outputs: array<f32, L1_SIZE>;
    var l2_outputs: array<f32, L2_SIZE>;

    // Layer 1: Input -> Hidden 1
    for (var i = 0u; i < L1_SIZE; i = i + 1u) {
        var sum = 0.0;
        for (var j = 0u; j < 8u; j = j + 1u) {
            sum = sum + input[j] * weights[i * INPUT_SIZE_WITH_BIAS + j];
        }
        sum = sum + weights[i * INPUT_SIZE_WITH_BIAS + 8u]; // Bias
        l1_outputs[i] = tanh(sum);
    }

    // Layer 2: Hidden 1 -> Hidden 2
    for (var i = 0u; i < L2_SIZE; i = i + 1u) {
        var sum = 0.0;
        for (var j = 0u; j < L1_SIZE; j = j + 1u) {
            sum = sum + l1_outputs[j] * weights[L1_WEIGHTS + i * L1_SIZE_WITH_BIAS + j];
        }
        sum = sum + weights[L1_WEIGHTS + i * L1_SIZE_WITH_BIAS + L1_SIZE]; // Bias
        l2_outputs[i] = tanh(sum);
    }

    // Layer 3: Hidden 2 -> Output
    var sum = 0.0;
    for (var j = 0u; j < L2_SIZE; j = j + 1u) {
        sum = sum + l2_outputs[j] * weights[L1_WEIGHTS + L2_WEIGHTS + j];
    }
    sum = sum + weights[L1_WEIGHTS + L2_WEIGHTS + L2_SIZE]; // Bias
    output[0] = sum;
}
