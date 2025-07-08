use crate::{config::Activation, constants::*, traits::Individual, utils};
use rand::Rng;

/// A neural network individual whose weights are stored in a heap-allocated `Vec<f32>`.
///
/// # Memory Layout
/// All weights (`TOTAL_WEIGHTS`) are stored in a `Vec<f32>`, which allocates its memory on the heap.
///
/// # Performance
/// - **Pros**: Flexible. The size of the network is not limited by the stack and can be determined
///   at runtime (though it's fixed by a constant in this project).
/// - **Cons**: Slightly slower than `StackIndividual` due to the overhead of heap allocation
///   and potential for worse cache locality if the memory is fragmented.
///
/// # Teaching Note
/// This struct is a direct contrast to `StackIndividual`. It showcases the trade-offs between
/// the stack and the heap. While the heap offers flexibility, it comes with a performance cost.
/// For this project, where the network size is small and fixed, the stack is superior, but the
/// heap would be necessary for larger, more complex models.
#[derive(Clone)]
pub struct HeapIndividual {
    /// All weights for the network, stored contiguously on the heap.
    pub weights: Vec<f32>,
}

impl Default for HeapIndividual {
    /// Creates a `HeapIndividual` with weights initialized to random values in `[-1, 1]`.
    ///
    /// # Teaching Note
    /// The `Default` trait is used by `Population::new` to create the initial population.
    /// Initializing with random weights is crucial for a genetic algorithm to ensure that
    /// the starting population has diversity, providing a wide base for evolution to begin.
    fn default() -> Self {
        let mut weights = vec![0.0; TOTAL_WEIGHTS];
        let mut rng = rand::rng();
        for weight in weights.iter_mut() {
            *weight = rng.random_range(-1.0..=1.0);
        }
        Self { weights }
    }
}

impl Individual for HeapIndividual {
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
