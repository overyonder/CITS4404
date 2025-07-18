use macroquad::input::is_key_down;

use macroquad::input::KeyCode;

use crate::nn::Individual;

/// A controller is a function that takes a gamestate and returns a decision.
pub trait Controller {
    fn pass(&self, state: &[f32; 9]) -> f32;
}

/// A player is a controller that uses keyboard input to move the paddle.
pub struct Player {
    pub up_key: KeyCode,
    pub down_key: KeyCode,
}

impl Controller for Player {
    fn pass(&self, _: &[f32; 9]) -> f32 {
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
    fn pass(&self, state: &[f32; 9]) -> f32 {
        let weights = self.get_weights();

        // First hidden layer (16 neurons)
        let mut layer2 = [0.; 17];
        for i in 0..16 {
            let start = i * 9;
            let end = start + 9;
            layer2[i] = state
                .iter()
                .zip(&weights[start..end])
                .map(|(a, w)| a * w)
                .sum::<f32>()
                // Apply ReLU activation function
                .max(0.0);
        }
        layer2[16] = 1.0;

        // Second hidden layer (4 neurons)
        let mut layer3 = [0.; 5];
        for i in 0..4 {
            let start = 144 + i * 17;
            let end = start + 17;
            layer3[i] = layer2
                .iter()
                .zip(&weights[start..end])
                .map(|(a, w)| a * w)
                .sum::<f32>()
                // Apply ReLU activation function
                .max(0.0);
        }
        layer3[4] = 1.0;

        // Output layer (1 neuron)
        let output = layer3
            .iter()
            .zip(&weights[212..217])
            .map(|(a, w)| a * w)
            .sum::<f32>();

        // Apply tanh activation function
        output.tanh()
    }
}
