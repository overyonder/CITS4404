// Game constants
pub const WIDTH: u16 = 1000;
pub const LENGTH: u16 = 1000;
pub const PADDLE_WIDTH: u16 = 1;
pub const PADDLE_HEIGHT: u16 = 100;
pub const PADDLE_MAX_VEL: f32 = 20.0;
pub const MAX_SCORE: u8 = 10;
pub const TICK_RATE: u16 = 60;
// Min and max for the PADDLE'S CENTER.
pub const MIN_PADDLE_POS: f32 = PADDLE_HEIGHT as f32 / 2.0;
pub const MAX_PADDLE_POS: f32 = LENGTH as f32 - PADDLE_HEIGHT as f32 / 2.0;

// Neural Network Layout
pub const LAYERS: [usize; 4] = [6, 16, 4, 1]; // Input, Hidden 1, Hidden 2, Output
pub const INPUT_SIZE: usize = LAYERS[0];
pub const L1_SIZE: usize = LAYERS[1];
pub const L2_SIZE: usize = LAYERS[2];
pub const L3_SIZE: usize = LAYERS[3];

// Calculate weights for each layer (including bias node)
pub const L1_WEIGHTS: usize = L1_SIZE * (INPUT_SIZE + 1);
pub const L2_WEIGHTS: usize = L2_SIZE * (L1_SIZE + 1);
pub const L3_WEIGHTS: usize = L3_SIZE * (L2_SIZE + 1);
pub const TOTAL_WEIGHTS: usize = L1_WEIGHTS + L2_WEIGHTS + L3_WEIGHTS;

// Genetic Algorithm constants
pub const POPULATION_SIZE: usize = 128;
pub const ELITE_COUNT: usize = 11; // ~sqrt(128)
pub const MUTATION_RATE: f32 = 0.1;
pub const MUTATION_STRENGTH: f32 = 0.2;

// File for saving the best network
pub const BEST_NET_FILE: &str = "best_net.bin";

