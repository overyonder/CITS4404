use crate::constants::{LAYERS, TOTAL_WEIGHTS};
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Net {
    pub weights: Vec<f64>,
}

impl Net {
    pub fn new() -> Self {
        // We use a fixed seed for deterministic initialization.
        let mut rng = StdRng::from_seed([0; 32]);
        let mut weights = Vec::with_capacity(TOTAL_WEIGHTS);
        for _ in 0..TOTAL_WEIGHTS {
            weights.push(rng.gen_range(-1.0..1.0));
        }
        Self { weights }
    }

    /// Creates a new network by combining the weights of two parent networks.
    pub fn crossover(p1: &Self, p2: &Self) -> Self {
        let mut rng = thread_rng();
        let mut child_weights = Vec::with_capacity(TOTAL_WEIGHTS);

        for i in 0..TOTAL_WEIGHTS {
            if rng.gen() { // Randomly pick between true and false
                child_weights.push(p1.weights[i]);
            } else {
                child_weights.push(p2.weights[i]);
            }
        }

        Self { weights: child_weights }
    }

    /// Applies a mutation to the network's weights.
    /// A single weight is chosen at random and perturbed by a value from a normal distribution.
    pub fn mutate(&mut self) {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let weight_index = rng.gen_range(0..self.weights.len());
        self.weights[weight_index] += normal.sample(&mut rng);
    }

    /// Performs a forward pass through the neural network.
    pub fn forward_propagate(&self, inputs: &[f64; 8]) -> f64 {
        assert_eq!(inputs.len(), LAYERS[0]);

        let mut current_activations: Vec<f64> = inputs.to_vec();
        let mut weight_offset = 0;

        // Iterate through each layer transition (e.g., 0->1, 1->2, 2->3)
        for i in 0..(LAYERS.len() - 1) {
            let input_size = LAYERS[i] + 1; // +1 for bias
            let output_size = LAYERS[i + 1];

            let mut next_activations = vec![0.0; output_size];

            for j in 0..output_size {
                let weights_start = weight_offset + j * input_size;
                let weights_end = weights_start + input_size;
                let weights_for_neuron = &self.weights[weights_start..weights_end];

                let mut sum = 0.0;
                // Add weighted inputs
                for k in 0..LAYERS[i] {
                    sum += current_activations[k] * weights_for_neuron[k];
                }
                // Add bias
                sum += weights_for_neuron[LAYERS[i]];

                // Apply activation function
                if i < LAYERS.len() - 2 {
                    // Hidden layers use ReLU
                    next_activations[j] = sum.max(0.0);
                } else {
                    // Output layer uses tanh to constrain output between -1 and 1
                    next_activations[j] = sum.tanh();
                }
            }

            current_activations = next_activations;
            weight_offset += input_size * output_size;
        }

        // The network has a single output neuron
        current_activations[0]
    }
}
