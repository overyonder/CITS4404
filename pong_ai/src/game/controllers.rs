use macroquad::input::is_key_down;

use macroquad::input::KeyCode;

use crate::nn::Individual;

/// A controller is a function that takes a gamestate and returns a decision.
pub trait Controller {
    fn pass(&self, state: &[f32; 8]) -> f32;
}

/// A player is a controller that uses keyboard input to move the paddle.
pub struct Player {
    pub up_key: KeyCode,
    pub down_key: KeyCode,
}

impl Controller for Player {
    fn pass(&self, _: &[f32; 8]) -> f32 {
        if is_key_down(self.up_key) {
            -1.
        } else if is_key_down(self.down_key) {
            1.
        } else {
            0.
        }
    }
}

impl Controller for Individual {
    fn pass(&self, state: &[f32; 8]) -> f32 {
        let weights = &self.weights;

        // First hidden layer (16 neurons)
        let mut layer1 = [0.0; 16];
        for i in 0..16 {
            let start = i * 8;
            let end = start + 8;
            let bias = weights[128 + i];
            layer1[i] = state
                .iter()
                .zip(&weights[start..end])
                .map(|(a, w)| a * w)
                .sum::<f32>()
                + bias;
            // Apply ReLU activation function
            layer1[i] = layer1[i].max(0.0);
        }

        // Second hidden layer (4 neurons)
        let mut layer2 = [0.0; 4];
        for i in 0..4 {
            let start = 144 + i * 16;
            let end = start + 16;
            let bias = weights[208 + i];
            layer2[i] = layer1
                .iter()
                .zip(&weights[start..end])
                .map(|(a, w)| a * w)
                .sum::<f32>()
                + bias;
            // Apply ReLU activation function
            layer2[i] = layer2[i].max(0.0);
        }

        // Output layer (1 neuron)
        let start = 212;
        let end = start + 4;
        let bias = weights[216];
        let output = layer2
            .iter()
            .zip(&weights[start..end])
            .map(|(a, w)| a * w)
            .sum::<f32>()
            + bias;

        // Apply tanh activation function
        output.tanh()
    }
}
