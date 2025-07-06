//! A neural network engine where all weights are stored in a single, stack-allocated array.

use crate::{config::Activation, constants::*, traits::Individual, utils};
use bytemuck;
use rand::Rng;
use std::io::Read;

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

    /// Loads a `StackIndividual` and its configuration from a file.
    ///
    /// # File Format
    /// The function expects the file to be in the format created by the `save` method:
    /// 1. `u64` (little-endian): Length of the JSON config string.
    /// 2. `[u8]`: The UTF-8 encoded JSON config string.
    /// 3. `[f32]`: The raw `f32` weights, which are read directly into the stack-allocated array.
    ///
    /// # Returns
    /// A `Result` containing a tuple of the loaded `StackIndividual` and its `Config`,
    /// or an error if reading or deserialization fails.
    ///
    /// # Teaching Note: Safe and Efficient Deserialization
    /// This function demonstrates a safe and highly efficient way to deserialize raw bytes
    /// back into structured data. Instead of reading floats one by one, it reads the entire
    /// block of weight data and then uses `bytemuck::cast_slice`. This is a zero-copy cast
    /// that reinterprets the byte slice as a float slice, which is safe because `f32` has
    /// no internal padding and a standard memory layout. The final `copy_from_slice` then
    /// efficiently moves the data into the new individual's stack-allocated array.
    fn load(path: &str) -> Result<(Self, crate::config::Config), Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let mut file = std::fs::File::open(path)?;

        // 1. Read config length
        let mut config_len_bytes = [0u8; 8];
        file.read_exact(&mut config_len_bytes)?;
        let config_len = u64::from_le_bytes(config_len_bytes);

        // 2. Read and deserialize config
        let mut config_bytes = vec![0u8; config_len as usize];
        file.read_exact(&mut config_bytes)?;
        let config: crate::config::Config = serde_json::from_slice(&config_bytes)?;

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

        let weights_slice: &[f32] = bytemuck::cast_slice(&weights_bytes);

        let mut individual = Self {
            weights: [0.0; TOTAL_WEIGHTS],
        };
        individual.weights.copy_from_slice(weights_slice);

        Ok((individual, config))
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
