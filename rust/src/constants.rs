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

/// The probability that any single weight in a network will be mutated.
pub const MUTATION_RATE: f64 = 0.05; // 5% chance

/// The maximum amount a weight can be changed during mutation.
pub const MUTATION_AMOUNT: f64 = 0.1;

/// The number of networks in each generation.
pub const POPULATION_SIZE: usize = 100;

/// The percentage of the best networks to carry over to the next generation.
pub const ELITISM_RATIO: f64 = 0.1; // Keep top 10%
pub const LAYERS: [usize; 4] = [8, 16, 4, 1];
pub const TOTAL_WEIGHTS: usize =
    (LAYERS[0] + 1) * LAYERS[1] + (LAYERS[1] + 1) * LAYERS[2] + (LAYERS[2] + 1) * LAYERS[3];
