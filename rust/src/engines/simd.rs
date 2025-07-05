use crate::{config::EvolutionConfig, constants::*, traits::Individual};
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// An individual represented by weights stored on the stack, optimized with SIMD.
#[derive(Clone, Copy)]
pub struct SimdIndividual {
    pub weights: [f32; TOTAL_WEIGHTS],
}

// SIMD dot product. This function is unsafe because it uses CPU intrinsics.
// It also requires the target CPU to support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    assert_eq!(len, b.len());

    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 7 < len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
        i += 8;
    }

    // Horizontal sum of the vector
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
    let mut sum = sum_arr.iter().sum();

    // Handle remaining elements
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

// Fallback dot product for non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// This helper function dispatches to the correct dot product implementation.
#[inline]
fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // This is unsafe because we are calling a function with SIMD intrinsics
        // which requires the CPU to support AVX2 and FMA.
        // We have enabled this with a target_feature attribute on the function.
        unsafe { dot_simd(a, b) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64 architectures.
        dot_simd(a, b)
    }
}

impl Individual for SimdIndividual {
    fn name() -> &'static str {
        "SIMD"
    }

    fn forward(&self, input: &[f32; INPUT_SIZE], _config: &EvolutionConfig) -> [f32; OUTPUT_SIZE] {
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
            l1_outputs[i] = (dispatch_dot(input, weights_slice) + bias).tanh();
        }

        // Layer 2: Hidden 1 -> Hidden 2
        for i in 0..HIDDEN2_SIZE {
            let start = i * (HIDDEN1_SIZE + 1);
            let end = start + HIDDEN1_SIZE;
            let weights_slice = &l2_weights[start..end];
            let bias = l2_weights[end];
            l2_outputs[i] = (dispatch_dot(&l1_outputs, weights_slice) + bias).tanh();
        }

        // Layer 3: Hidden 2 -> Output
        for i in 0..OUTPUT_SIZE {
            let start = i * (HIDDEN2_SIZE + 1);
            let end = start + HIDDEN2_SIZE;
            let weights_slice = &l3_weights[start..end];
            let bias = l3_weights[end];
            output[i] = dispatch_dot(&l2_outputs, weights_slice) + bias;
        }

        output
    }

    fn weights_as_slice(&self) -> &[f32] {
        &self.weights
    }


    fn recombine_from<R: Rng>(
        &mut self,
        p1: &Self,
        p2: &Self,
        rng: &mut R,
        config: &EvolutionConfig,
    ) {
        let normal = Normal::new(0.0, config.mutation_strength).unwrap();
        for i in 0..TOTAL_WEIGHTS {
            self.weights[i] = if rng.gen::<bool>() {
                p1.weights[i]
            } else {
                p2.weights[i]
            };
            if rng.gen::<f32>() < config.mutation_rate {
                self.weights[i] += normal.sample(rng);
            }
        }
    }
}

impl Default for SimdIndividual {
    fn default() -> Self {
        let mut weights = [0.0; TOTAL_WEIGHTS];
        let mut rng = rand::thread_rng();
        for weight in weights.iter_mut() {
            *weight = rng.gen_range(-1.0..=1.0);
        }
        Self { weights }
    }
}
