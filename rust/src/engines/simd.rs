//! A neural network engine that uses SIMD (Single Instruction, Multiple Data) for acceleration.

use crate::{config::Activation, constants::*, traits::Individual, utils};
use rand::Rng;
use std::io::Read;
use bytemuck;

// Use architecture-specific intrinsics for x86_64.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A neural network individual that uses SIMD instructions for an optimized forward pass.
///
/// # Memory Layout
/// Like `StackIndividual`, this struct stores all weights in a single, stack-allocated array
/// to ensure cache-friendliness and avoid heap allocations.
///
/// # Performance
/// The key difference is the use of SIMD intrinsics (specifically AVX2 and FMA) for the
/// dot product calculation, which is the most computationally intensive part of the forward pass.
/// This allows the CPU to perform multiple floating-point multiplications and additions in a
/// single instruction, offering a significant speedup over the scalar `StackIndividual`.
///
/// # Teaching Note
/// This is a great example of performance optimization via architecture-specific features.
/// The use of `#[cfg]` and `#[target_feature]` allows the code to compile on any platform,
/// but it will only use the SIMD optimizations when targeting a compatible `x86_64` CPU.
#[derive(Clone, Copy, Debug)]
pub struct SimdIndividual {
    pub weights: [f32; TOTAL_WEIGHTS],
}

/// Performs a dot product using AVX2 and FMA intrinsics for high performance.
///
/// # Safety
/// This function is `unsafe` because it directly calls CPU intrinsics that are not
/// guaranteed to be available on all hardware. The `#[target_feature]` attribute ensures
/// the compiler only generates this code when AVX2 and FMA are enabled, but the call
/// itself remains `unsafe` to signal this dependency to the programmer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    assert_eq!(len, b.len(), "Slices must have the same length for dot product.");

    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats (256 bits) at a time.
    while i + 7 < len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        // Fused Multiply-Add: sum_vec = (a_vec * b_vec) + sum_vec
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
        i += 8;
    }

    // Horizontally sum the elements in the 256-bit vector.
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
    let mut sum = sum_arr.iter().sum();

    // Handle any remaining elements that didn't fit into a full SIMD vector.
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

/// A fallback scalar dot product for non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Dispatches to the correct dot product implementation based on the target architecture.
///
/// # Teaching Note: Conditional Compilation
/// This function is a key part of making the SIMD engine portable. It uses `#[cfg]`
/// attributes to conditionally compile code. On `x86_64` targets, it calls the `unsafe`
/// `dot_simd` function. On all other architectures, it calls the safe, scalar fallback
/// implementation. This ensures the program compiles and runs everywhere, while only
/// enabling the high-performance SIMD code where it's supported.
#[inline]
fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 {
    // On x86_64, call the unsafe SIMD version. On other platforms, use the safe fallback.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        dot_simd(a, b)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_simd(a, b)
    }
}

impl Individual for SimdIndividual {
    fn forward_propagate(&self, input: &[f32; INPUT_SIZE], activation: Activation) -> [f32; OUTPUT_SIZE] {
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
                let sum = dispatch_dot(input, weights) + bias;
                *out = utils::apply_activation(sum, activation);
            });

        // Layer 2: Hidden 1 -> Hidden 2
        l2_weights
            .chunks_exact(HIDDEN1_SIZE + 1)
            .zip(l2_outputs.iter_mut())
            .for_each(|(neuron_weights, out)| {
                let (weights, bias_slice) = neuron_weights.split_at(HIDDEN1_SIZE);
                let bias = bias_slice[0];
                let sum = dispatch_dot(&l1_outputs, weights) + bias;
                *out = utils::apply_activation(sum, activation);
            });

        // Layer 3: Hidden 2 -> Output
        l3_weights
            .chunks_exact(HIDDEN2_SIZE + 1)
            .zip(output.iter_mut())
            .for_each(|(neuron_weights, out)| {
                let (weights, bias_slice) = neuron_weights.split_at(HIDDEN2_SIZE);
                let bias = bias_slice[0];
                *out = dispatch_dot(&l2_outputs, weights) + bias;
            });

        output
    }

    fn weights_as_slice(&self) -> &[f32] {
        &self.weights
    }

    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Loads a `SimdIndividual` and its configuration from a file.
    ///
    /// # File Format
    /// The function expects the file to be in the format created by the `save` method:
    /// 1. `u64` (little-endian): Length of the JSON config string.
    /// 2. `[u8]`: The UTF-8 encoded JSON config string.
    /// 3. `[f32]`: The raw `f32` weights, which are read directly into the stack-allocated array.
    ///
    /// # Returns
    /// A `Result` containing a tuple of the loaded `SimdIndividual` and its `Config`,
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



impl Default for SimdIndividual {
    /// Creates a `SimdIndividual` with weights initialized to random values in `[-1, 1]`.
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
