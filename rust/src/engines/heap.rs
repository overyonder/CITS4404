use crate::{config::Activation, constants::*, traits::Individual, utils};
use bytemuck;
use rand::Rng;
use std::io::Read;

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

    /// Loads a `HeapIndividual` and its configuration from a file.
    ///
    /// # File Format
    /// The function expects the file to be in the format created by the `save` method:
    /// 1. `u64` (little-endian): Length of the JSON config string.
    /// 2. `[u8]`: The UTF-8 encoded JSON config string.
    /// 3. `[f32]`: The raw `f32` weights, which are read into a new heap-allocated `Vec<f32>`.
    ///
    /// # Returns
    /// A `Result` containing a tuple of the loaded `HeapIndividual` and its `Config`,
    /// or an error if reading or deserialization fails.
    ///
    /// # Teaching Note: Deserialization to the Heap
    /// The process is similar to the `StackIndividual`, but with a key difference. After
    /// `bytemuck::cast_slice` provides a temporary, zero-copy view of the bytes as `&[f32]`,
    /// `.to_vec()` is called. This performs a new heap allocation and copies the weight
    /// data into it, creating the `Vec<f32>` that the `HeapIndividual` owns.
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

        let weights: Vec<f32> = bytemuck::cast_slice(&weights_bytes).to_vec();

        let individual = Self { weights };

        Ok((individual, config))
    }
}
