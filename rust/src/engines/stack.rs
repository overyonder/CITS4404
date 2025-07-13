//! A neural network engine where all weights are stored in a single, stack-allocated array.

use crate::{config::Activation, engines::{constants::*, utils}, traits::Individual};
use rand::Rng;

/// A neural network individual using a stack-allocated, fixed-size array for its weights.
///
/// # Memory Layout
/// All weights (`TOTAL_WEIGHTS`) are stored in a single `[f32; N]` array directly within the struct.
/// When a `StackIndividual` is created, it resides entirely on the call stack.
///
/// # Performance
/// - **Pros**: Extremely fast. No heap allocation means no overhead from the system allocator.
///   Excellent cache locality, as all weights are contiguous in memory.
/// - **Cons**: The size of the network is fixed at compile time and is limited by the stack size,
///   which is typically much smaller than the heap.
///
/// # Teaching Note
/// This struct is a perfect example of leveraging the stack for performance. It's the most
/// efficient implementation possible for a network of this size, but it lacks the flexibility
/// of heap-based approaches for larger, dynamically-sized networks.
#[derive(Clone, Copy, Debug)]
pub struct StackIndividual {
    /// All weights for the network, stored contiguously on the stack.
    pub weights: [f32; TOTAL_WEIGHTS],
}

impl Individual for StackIndividual {
    /// Performs a forward pass using manually sliced portions of the flat weight array.
    ///
    /// # Algorithm
    /// 1.  The flat `weights` array is split into chunks for each layer.
    /// 2.  For each neuron in a layer, its inputs are multiplied by their corresponding weights.
    /// 3.  The weighted sum is added to a bias term.
    /// 4.  The result is passed through an activation function.
    /// 5.  The output of one layer becomes the input for the next.
    ///
    /// # Teaching Note
    /// This is a manual implementation of matrix multiplication. It demonstrates how a neural
    /// network, often visualized as interconnected layers, is actually implemented under the
    /// hood using flat arrays and loops. Understanding this mapping is key to creating
    /// efficient neural network engines.
    fn forward_propagate(
        &self,
        input: &[f32; INPUT_SIZE],
        activation: Activation,
    ) -> [f32; OUTPUT_SIZE] {
        let mut l1_outputs = [0.0; HIDDEN1_SIZE];
        let mut l2_outputs = [0.0; HIDDEN2_SIZE];
        let mut output = [0.0; OUTPUT_SIZE];

        let (l1_weights, rest) = self.weights.split_at(L1_WEIGHTS);
        let (l2_weights, l3_weights) = rest.split_at(L2_WEIGHTS);

        // Layer 1: Input -> Hidden 1
        l1_weights
            .chunks_exact(INPUT_SIZE + 1)
            .zip(l1_outputs.iter_mut())
            .for_each(|(neuron_weights, out)| {
                let (weights, bias_slice) = neuron_weights.split_at(INPUT_SIZE);
                let bias = bias_slice[0];
                let sum = utils::dot(input, weights) + bias;
                *out = utils::apply_activation(sum, activation);
            });

        // Layer 2: Hidden 1 -> Hidden 2
        l2_weights
            .chunks_exact(HIDDEN1_SIZE + 1)
            .zip(l2_outputs.iter_mut())
            .for_each(|(neuron_weights, out)| {
                let (weights, bias_slice) = neuron_weights.split_at(HIDDEN1_SIZE);
                let bias = bias_slice[0];
                let sum = utils::dot(&l1_outputs, weights) + bias;
                *out = utils::apply_activation(sum, activation);
            });

        // Layer 3: Hidden 2 -> Output
        l3_weights
            .chunks_exact(HIDDEN2_SIZE + 1)
            .zip(output.iter_mut())
            .for_each(|(neuron_weights, out)| {
                let (weights, bias_slice) = neuron_weights.split_at(HIDDEN2_SIZE);
                let bias = bias_slice[0];
                *out = utils::dot(&l2_outputs, weights) + bias;
            });

        output
    }

    fn weights_as_slice(&self) -> &[f32] {
        &self.weights
    }

    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

impl Default for StackIndividual {
    /// Creates a `StackIndividual` with weights initialized to random values in `[-1, 1]`.
    ///
    /// # Teaching Note
    /// The `Default` trait is used by `Population::new` to create the initial population.
    /// Initializing with random weights is crucial for a genetic algorithm to ensure that
    /// the starting population has diversity, providing a wide base for evolution to begin.
    /// If they were all initialized to zero, they would all behave identically.
    fn default() -> Self {
        let mut weights = [0.0; TOTAL_WEIGHTS];
        let mut rng = rand::rng();
        for weight in weights.iter_mut() {
            *weight = rng.random_range(-1.0..=1.0);
        }
        Self { weights }
    }
}
