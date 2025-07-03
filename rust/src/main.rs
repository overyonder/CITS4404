use rand::Rng;
use std::fmt;
use std::simd::f64x16;
use std::simd::f64x4;
use std::simd::f64x8;

const MAX_SCORE: u8 = 1;
const TICK_RATE: u16 = 60;
const LENGTH: u16 = 400;
const WIDTH: u16 = 300;
const PADDLE_WIDTH: u16 = WIDTH / 8;
const PADDLE_MAX_VEL: u16 = WIDTH / TICK_RATE;
const MAX_POSITION: u16 = WIDTH - PADDLE_WIDTH;
const BALL_START_VEL: [i16; 2] = [
    LENGTH as i16 / TICK_RATE as i16,
    LENGTH as i16 / TICK_RATE as i16,
];
const LAYERS: &[usize] = &[8, 16, 4, 1];
const TOTAL_WEIGHTS: usize =
    (LAYERS[0] + 1) * LAYERS[1] + (LAYERS[1] + 1) * LAYERS[2] + (LAYERS[2] + 1) * LAYERS[3];

struct Net {
    weights: [f64; TOTAL_WEIGHTS],
}

struct GameState {
    ball_position: [u16; 2], // [0, WIDTH] x [0, LENGTH]
    ball_velocity: [i16; 2], // [-PADDLE_MAX_VEL, PADDLE_MAX_VEL]
    left_position: u16,      // [0, WIDTH - PADDLE_WIDTH]
    left_velocity: i16,      // [-PADDLE_MAX_VEL, PADDLE_MAX_VEL]
    right_position: u16,     // [0, WIDTH - PADDLE_WIDTH]
    right_velocity: i16,     // [-PADDLE_MAX_VEL, PADDLE_MAX_VEL]
    left_score: u8,
    right_score: u8,
    left_returns: u16,
    right_returns: u16,
    left_shots: u16,
    right_shots: u16,
}

impl GameState {
    fn update_paddles(&mut self, left: &Net, right: &Net) {
        self.left_velocity = left.update_and_respond(self);
        self.right_velocity = right.update_and_respond(self);
        self.left_position = self.left_position.saturating_add_signed(self.left_velocity);
        self.right_position = self
            .right_position
            .saturating_add_signed(self.right_velocity);
        if self.left_position > MAX_POSITION {
            self.left_position = MAX_POSITION
        }
        if self.right_position > MAX_POSITION {
            self.right_position = MAX_POSITION
        }
    }

    fn update_ball(&mut self) {}

    fn tick(&mut self) {}
}

impl fmt::Debug for Net {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Net")
            .field("weights", &self.weights)
            .finish()
    }
}

impl Net {
    fn new() -> Net {
        let mut rng = rand::rng();
        Net {
            weights: rng.random(),
        }
    }

    fn update_and_respond(&self, state: &GameState) -> i16 {
        // Input layer
        let input = [
            state.ball_position[0] as f64, // 0 to WIDTH
            state.ball_position[1] as f64, // 0 to LENGTH
            state.ball_velocity[0] as f64, // -PADDLE_MAX_VEL to PADDLE_MAX_VEL
            state.ball_velocity[1] as f64, // -PADDLE_MAX_VEL to PADDLE_MAX_VEL
            state.left_position as f64,    // 0 to WIDTH
            state.left_velocity as f64,    // -PADDLE_MAX_VEL to PADDLE_MAX_VEL
            state.right_position as f64,   // 0 to WIDTH
            state.right_velocity as f64,   // -PADDLE_MAX_VEL to PADDLE_MAX_VEL
        ];

        // Layer 1: 8 inputs → 16 neurons
        let mut layer1 = [0.0; 16];
        let mut weight_idx = 0;
        for i in 0..16 {
            for j in 0..8 {
                layer1[i] += self.weights[weight_idx] * input[j];
                weight_idx += 1;
            }
            layer1[i] += self.weights[weight_idx]; // Bias
            weight_idx += 1;
            layer1[i] = layer1[i].clamp(-1.0, 1.0); // Activation function
        }

        // Layer 2: 16 inputs → 4 neurons
        let mut layer2 = [0.0; 4];
        for i in 0..4 {
            for j in 0..16 {
                layer2[i] += self.weights[weight_idx] * layer1[j];
                weight_idx += 1;
            }
            layer2[i] += self.weights[weight_idx]; // Bias
            weight_idx += 1;
            layer2[i] = layer2[i].clamp(-1.0, 1.0); // Activation function
        }

        // Layer 3: 4 inputs → 1 neuron
        let mut output = 0.0;
        for i in 0..4 {
            output += self.weights[weight_idx] * layer2[i];
            weight_idx += 1;
        }
        output += self.weights[weight_idx]; // Bias
        output = output.clamp(-1.0, 1.0); // Activation function

        // Final output clamping
        (output * PADDLE_MAX_VEL as f64).clamp(-PADDLE_MAX_VEL as f64, PADDLE_MAX_VEL as f64) as i16
    }

    fn mutate(&self) {}

    fn evolve(&self) {
        self.mutate();
    }
}

fn main() {
    let net = Net::new();
    // net.evolve();
    // println!("{:?}", net);
}
