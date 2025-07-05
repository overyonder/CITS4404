// Game constants
pub const WIDTH: u16 = 1000;
pub const LENGTH: u16 = 1000;
pub const PADDLE_WIDTH: u16 = 1;
pub const PADDLE_HEIGHT: u16 = 100;
pub const PADDLE_MAX_VEL: f32 = 20.0;
pub const MAX_SCORE: u8 = 10;
pub const TICK_RATE: u16 = 60;
// Min and max for the PADDLE'S CENTER.
/// The minimum position of the paddle's center along the y-axis.
pub const MIN_PADDLE_POS: f32 = PADDLE_HEIGHT as f32 / 2.0;
/// The maximum position of the paddle's center along the y-axis.
pub const MAX_PADDLE_POS: f32 = LENGTH as f32 - PADDLE_HEIGHT as f32 / 2.0;

// Neural Network Layout
/// The structure of the neural network, represented as an array of layer sizes.
pub const LAYERS: [usize; 4] = [8, 16, 4, 1]; // Input, Hidden 1, Hidden 2, Output
/// Number of input neurons to the neural network (game state: ball/paddle positions & velocities).
pub const INPUT_SIZE: usize = LAYERS[0];
/// Number of neurons in the first hidden layer.
pub const HIDDEN1_SIZE: usize = LAYERS[1];
/// Number of neurons in the second hidden layer.
pub const HIDDEN2_SIZE: usize = LAYERS[2];
/// Number of output neurons (paddle command).
pub const OUTPUT_SIZE: usize = LAYERS[3];

// Calculate weights for each layer (including bias node)
/// The number of weights in the first layer, including the bias node.
pub const L1_WEIGHTS: usize = HIDDEN1_SIZE * (INPUT_SIZE + 1);
/// The number of weights in the second layer, including the bias node.
pub const L2_WEIGHTS: usize = HIDDEN2_SIZE * (HIDDEN1_SIZE + 1);
/// The number of weights in the third layer, including the bias node.
pub const L3_WEIGHTS: usize = OUTPUT_SIZE * (HIDDEN2_SIZE + 1);
/// The total number of weights in the neural network.
pub const TOTAL_WEIGHTS: usize = L1_WEIGHTS + L2_WEIGHTS + L3_WEIGHTS;

// GA parameters
/// Number of individuals (neural nets) per generation.
/// Higher = more diversity, but slower evaluation.
pub const POPULATION_SIZE: usize = 128;
/// Number of top individuals preserved unchanged each generation.
/// Higher = more stability, lower = more turnover.
pub const ELITE_COUNT: usize = 11; // ~sqrt(128)
