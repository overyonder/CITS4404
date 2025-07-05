use crate::{config::Activation, constants::*, traits::Individual, utils};
use rand::Rng;

/// An individual represented by weights stored on the heap.
#[derive(Clone)]
/// A neural network individual whose weights are stored on the heap (Vec).
///
/// # Memory Layout
/// - All weights are stored in a heap-allocated `Vec<f32>`.
/// - Allows for flexible, dynamic allocation of weights.
///
/// # Teaching Note
/// - Demonstrates heap allocation and dynamic memory management in Rust.
pub struct HeapIndividual {
    /// All weights for the network, stored on the heap.
    pub weights: Vec<f32>,
}

impl Default for HeapIndividual {
    fn default() -> Self {
        let mut weights = vec![0.0; TOTAL_WEIGHTS];
        let mut rng = rand::thread_rng();
        for weight in weights.iter_mut() {
            *weight = rng.gen_range(-1.0..=1.0);
        }
        Self { weights }
    }
}

impl Individual for HeapIndividual {
    fn name() -> &'static str {
        "Heap"
    }

    fn forward_propagate(&self, input: &[f32; INPUT_SIZE], activation: Activation) -> [f32; OUTPUT_SIZE] {
        let mut l1_outputs = [0.0; HIDDEN1_SIZE];
        let mut l2_outputs = [0.0; HIDDEN2_SIZE];
        let mut output = [0.0; OUTPUT_SIZE];

        let (l1_weights, rest) = self.weights.split_at(L1_WEIGHTS);
        let (l2_weights, l3_weights) = rest.split_at(L2_WEIGHTS);

        // Layer 1: Input -> Hidden 1
        for i in 0..HIDDEN1_SIZE {
            let start = i * (INPUT_SIZE + 1);
            let end = start + INPUT_SIZE;
            let weights_slice = &l1_weights[start..end];
            let bias = l1_weights[end];
            let sum = utils::dot(input, weights_slice) + bias;
            l1_outputs[i] = utils::apply_activation(sum, activation);
        }

        // Layer 2: Hidden 1 -> Hidden 2
        for i in 0..HIDDEN2_SIZE {
            let start = i * (HIDDEN1_SIZE + 1);
            let end = start + HIDDEN1_SIZE;
            let weights_slice = &l2_weights[start..end];
            let bias = l2_weights[end];
            let sum = utils::dot(&l1_outputs, weights_slice) + bias;
            l2_outputs[i] = utils::apply_activation(sum, activation);
        }

        // Layer 3: Hidden 2 -> Output (No activation on the output layer)
        for i in 0..OUTPUT_SIZE {
            let start = i * (HIDDEN2_SIZE + 1);
            let end = start + HIDDEN2_SIZE;
            let weights_slice = &l3_weights[start..end];
            let bias = l3_weights[end];
            output[i] = utils::dot(&l2_outputs, weights_slice) + bias;
        }

        output
    }

    fn weights_as_slice(&self) -> &[f32] {
        &self.weights
    }

    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}
