use crate::config::EvolutionConfig;
use crate::constants::*;
use crate::traits::Individual;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// An individual represented by weights stored on the stack.
#[derive(Clone, Copy)]
pub struct StackIndividual {
    pub weights: [f32; TOTAL_WEIGHTS],
}

impl Individual for StackIndividual {
    fn name() -> &'static str {
        "Stack"
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
            l1_outputs[i] = (dot(input, weights_slice) + bias).tanh();
        }

        // Layer 2: Hidden 1 -> Hidden 2
        for i in 0..HIDDEN2_SIZE {
            let start = i * (HIDDEN1_SIZE + 1);
            let end = start + HIDDEN1_SIZE;
            let weights_slice = &l2_weights[start..end];
            let bias = l2_weights[end];
            l2_outputs[i] = (dot(&l1_outputs, weights_slice) + bias).tanh();
        }

        // Layer 3: Hidden 2 -> Output
        for i in 0..OUTPUT_SIZE {
            let start = i * (HIDDEN2_SIZE + 1);
            let end = start + HIDDEN2_SIZE;
            let weights_slice = &l3_weights[start..end];
            let bias = l3_weights[end];
            output[i] = dot(&l2_outputs, weights_slice) + bias;
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
            // Crossover
            self.weights[i] = if rng.gen::<bool>() { p1.weights[i] } else { p2.weights[i] };

            // Mutation
            if rng.gen::<f32>() < config.mutation_rate {
                self.weights[i] += normal.sample(rng);
            }
        }
    }
}

/// Computes the dot product of two slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Default for StackIndividual {
    fn default() -> Self {
        let mut weights = [0.0; TOTAL_WEIGHTS];
        let mut rng = rand::thread_rng();
        for weight in weights.iter_mut() {
            *weight = rng.gen_range(-1.0..=1.0);
        }
        Self { weights }
    }
}
