pub const WIDTH: u16 = 1000;
pub const LENGTH: u16 = 1000;
pub const PADDLE_WIDTH: u16 = 10;
pub const PADDLE_HEIGHT: u16 = 100;
pub const PADDLE_MAX_VEL: i16 = 10;
pub const MAX_SCORE: u8 = 10;
pub const TICK_RATE: u16 = 60;
pub const MAX_POSITION: u16 = LENGTH - PADDLE_HEIGHT;
pub const BALL_START_VEL: [f64; 2] = [
    (LENGTH as f64) / (TICK_RATE as f64),
    (LENGTH as f64) / (TICK_RATE as f64),
];

// --- Genetic Algorithm Constants ---

/// The number of networks in each generation.
pub const POPULATION_SIZE: usize = 128;

/// The number of the best networks to carry over to the next generation.
pub const ELITE_COUNT: usize = 11; // sqrt(128)
pub const LAYERS: [usize; 4] = [8, 16, 4, 1];

// A const function to calculate the total number of weights and biases needed.
const fn get_total_weights() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < LAYERS.len() - 1 {
        total += LAYERS[i] * LAYERS[i + 1] + LAYERS[i + 1]; // weights + biases
        i += 1;
    }
    total
}

pub const TOTAL_WEIGHTS: usize = get_total_weights();
