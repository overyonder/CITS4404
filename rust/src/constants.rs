//! Defines compile-time constants used throughout the application.
//!
//! This module centralizes all the magic numbers and configuration values that are
//! fixed at compile time. This improves maintainability, as critical values for the
//! simulation and neural network can be adjusted in one place.
//!
//! # Teaching Note
//! Using a dedicated `constants` module is a common and effective practice in Rust for
//! managing global, immutable values. The `pub const` keyword makes these values
//! publicly accessible and ensures they are inlined at compile time, incurring no
//! runtime overhead.

// ----------------------------------------------------------------------------
// Game World Constants
// ----------------------------------------------------------------------------

/// The width of the Pong game area in pixels.
pub const WIDTH: u16 = 1000;

/// The height of the Pong game area in pixels.
///
/// # Teaching Note
/// Using `WIDTH` and `HEIGHT` is a standard convention for 2D spaces, making the
/// code more intuitive to read than alternatives like `LENGTH`.
pub const HEIGHT: u16 = 1000;

/// The width of each paddle in pixels.
pub const PADDLE_WIDTH: u16 = 1;

/// The height of each paddle in pixels.
pub const PADDLE_HEIGHT: u16 = 100;

/// The maximum score before a game ends.
pub const MAX_SCORE: u8 = 10;

// ----------------------------------------------------------------------------
// Physics and Simulation Constants
// ----------------------------------------------------------------------------

/// The number of simulation ticks that occur per second.
///
/// # Teaching Note
/// This controls the simulation's temporal resolution. A higher `TICK_RATE` leads to
/// smoother motion and more accurate physics, but requires more computational power.
pub const TICK_RATE: u16 = 60;

/// The maximum velocity of a paddle in pixels per tick.
pub const PADDLE_MAX_VEL: f32 = 20.0;

/// The radius of the ball in pixels.
///
/// # Teaching Note
/// Using a radius allows for more accurate collision detection against the ball's edge
/// rather than its center point.
pub const BALL_RADIUS: f32 = 10.0;

/// The initial speed of the ball in pixels per tick.
pub const BALL_INITIAL_SPEED: f32 = 8.0;

/// The maximum speed the ball can reach.
///
/// # Teaching Note
/// Clamping the ball's speed is important for game balance, preventing it from
/// becoming uncontrollably fast after many paddle hits.
pub const BALL_MAX_SPEED: f32 = 25.0;

/// The minimum y-coordinate for a paddle's center.
/// This is calculated to be half the paddle's height, preventing any part of the
/// paddle from going above the top edge of the screen.
pub const MIN_PADDLE_POS: f32 = PADDLE_HEIGHT as f32 / 2.0;

/// The maximum y-coordinate for a paddle's center.
/// This is calculated to prevent any part of the paddle from going below the
/// bottom edge of the screen.
pub const MAX_PADDLE_POS: f32 = HEIGHT as f32 - PADDLE_HEIGHT as f32 / 2.0;

// ----------------------------------------------------------------------------
// Neural Network Architecture
// ----------------------------------------------------------------------------

/// The structure of the neural network, defined as an array of layer sizes.
/// The format is [Input, Hidden1, Hidden2, ..., Output].
pub const LAYERS: [usize; 4] = [8, 16, 4, 1];

/// Number of input neurons. These correspond to the game state variables fed to the network.
pub const INPUT_SIZE: usize = LAYERS[0];
/// Number of neurons in the first hidden layer.
pub const HIDDEN1_SIZE: usize = LAYERS[1];
/// Number of neurons in the second hidden layer.
pub const HIDDEN2_SIZE: usize = LAYERS[2];
/// Number of output neurons. This corresponds to the paddle's desired movement.
pub const OUTPUT_SIZE: usize = LAYERS[3];

/// The number of weights connecting the input layer to the first hidden layer.
///
/// # Teaching Note
/// The `+ 1` in `(INPUT_SIZE + 1)` accounts for the **bias neuron**. Each neuron in
/// the hidden layer receives input from all neurons in the previous layer, plus one
/// extra input from a virtual bias neuron that always outputs `1.0`. This bias weight
/// allows the neuron's activation function to be shifted left or right, increasing
/// the model's flexibility. The formula for weights in a fully connected layer is:
/// `num_weights = (num_inputs + 1) * num_outputs`.
pub const L1_WEIGHTS: usize = HIDDEN1_SIZE * (INPUT_SIZE + 1);

/// The number of weights connecting the first hidden layer to the second.
pub const L2_WEIGHTS: usize = HIDDEN2_SIZE * (HIDDEN1_SIZE + 1);

/// The number of weights connecting the second hidden layer to the output layer.
pub const L3_WEIGHTS: usize = OUTPUT_SIZE * (HIDDEN2_SIZE + 1);

/// The total number of weights (genes) in an individual's genome.
/// This is the sum of all weights in all layers and represents the total number of
/// parameters the genetic algorithm needs to optimize.
pub const TOTAL_WEIGHTS: usize = L1_WEIGHTS + L2_WEIGHTS + L3_WEIGHTS;
