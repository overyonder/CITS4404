use crate::constants::{LAYERS, MUTATION_AMOUNT, MUTATION_RATE, TOTAL_WEIGHTS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct Net {
    pub weights: [f64; TOTAL_WEIGHTS],
}

impl Net {
    pub fn new() -> Self {
        // We use a fixed seed for deterministic initialization.
        let mut rng = StdRng::from_seed([0; 32]);
        let mut weights = [0.0; TOTAL_WEIGHTS];
        for w in weights.iter_mut() {
            *w = rng.gen_range(-1.0..1.0);
        }
        Self { weights }
    }

    /// Activation function for a neuron. Clamps the value between -1.0 and 1.0.
    fn activation(x: f64) -> f64 {
        x.clamp(-1.0, 1.0)
    }

    /// Performs a forward pass through the neural network.
    pub fn forward_propagate(&self, inputs: &[f64; LAYERS[0]]) -> f64 {
        // The largest layer size is used for the ping-pong buffers.
        // This must be updated if LAYERS changes.
        const MAX_LAYER_SIZE: usize = 16;

        let mut values1 = [0.0; MAX_LAYER_SIZE];
        let mut values2 = [0.0; MAX_LAYER_SIZE];

        // `prev_layer_values` starts by pointing to the network's input slice.
        let mut prev_layer_values: &[f64] = inputs;
        let mut weight_idx = 0;

        // Iterate through each layer, starting from the first hidden layer.
        for i in 1..LAYERS.len() {
            let current_layer_size = LAYERS[i];
            let prev_layer_size = LAYERS[i - 1];

            // Determine which buffer to write to (ping-pong).
            let current_layer_buffer = if i % 2 == 1 {
                &mut values1[..current_layer_size]
            } else {
                &mut values2[..current_layer_size]
            };

            for neuron_idx in 0..current_layer_size {
                // Calculate the weighted sum of inputs from the previous layer.
                let mut sum = 0.0;
                for prev_neuron_idx in 0..prev_layer_size {
                    sum += prev_layer_values[prev_neuron_idx] * self.weights[weight_idx];
                    weight_idx += 1;
                }
                // Add the bias for the current neuron.
                sum += self.weights[weight_idx];
                weight_idx += 1;

                current_layer_buffer[neuron_idx] = Self::activation(sum);
            }
            // The buffer we just wrote to becomes the 'previous layer' for the next iteration.
            prev_layer_values = current_layer_buffer;
        }

        // The final result is in the last buffer we wrote to, which has a size of 1.
        prev_layer_values[0]
    }

    pub fn mutate(&mut self) {
        // Use a seeded RNG for deterministic mutations during training.
        // In a real application, you might use from_entropy() for more randomness.
        let mut rng = StdRng::from_seed([1; 32]);

        for weight in self.weights.iter_mut() {
            if rng.gen::<f64>() < MUTATION_RATE {
                let change = rng.gen_range(-MUTATION_AMOUNT..MUTATION_AMOUNT);
                *weight += change;
                // Clamp the weight to keep it within a reasonable range.
                *weight = weight.clamp(-1.0, 1.0);
            }
        }
    }

    pub fn evolve(&mut self) {
        self.mutate();
    }
}
