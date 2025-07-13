//! Neural network architecture constants for the Pong AI.
//!
//! These constants define the structure and parameters of the neural network
//! used by all engine implementations (CPU, GPU, etc.).

// ----------------------------------------------------------------------------
// Neural Network Architecture Constants
// ----------------------------------------------------------------------------

/// The structure of the neural network, defined as an array of layer sizes.
/// Format: `[Input, Hidden1, Hidden2, Output]`
///
/// # Teaching Note: Architecture Design Principles
/// This 4-layer architecture (8→16→4→1) represents several design decisions:
/// 
/// ## Input Layer (8 neurons):
/// Captures the **complete observable state** of the Pong game. Each input
/// corresponds to one piece of information the AI needs to make decisions.
/// 
/// ## Hidden Layer 1 (16 neurons):
/// **Feature extraction layer** - learns to detect useful patterns like:
/// - "Ball approaching my side"
/// - "Opponent moving upward"  
/// - "Ball will hit top wall soon"
/// The 2:1 expansion ratio (8→16) allows rich feature representation.
/// 
/// ## Hidden Layer 2 (4 neurons):
/// **Decision integration layer** - combines features into strategic decisions:
/// - "Should I move up?"
/// - "Should I stay centered?"
/// - "Should I intercept aggressively?"
/// The 4:1 compression ratio (16→4) forces abstraction and prioritization.
/// 
/// ## Output Layer (1 neuron):
/// **Motor control** - directly controls paddle velocity.
/// Single output simplifies the control problem to a 1D decision.
pub const LAYERS: [usize; 4] = [8, 16, 4, 1];

/// Number of input neurons corresponding to the game state variables:
/// 
/// # Teaching Note: State Representation Design
/// These 8 inputs provide the **minimal complete state** for optimal Pong play:
/// 
/// ## Own Paddle State (2 inputs):
/// 1. `own_paddle.y` - Current position for path planning
/// 2. `own_paddle.vy` - Current velocity for momentum awareness
/// 
/// ## Opponent Paddle State (2 inputs):
/// 3. `opponent_paddle.y` - Position for strategic aiming
/// 4. `opponent_paddle.vy` - Velocity for prediction of opponent moves
/// 
/// ## Ball State (4 inputs):
/// 5. `ball.x` - Position for time-to-collision calculations
/// 6. `ball.y` - Position for interception planning
/// 7. `ball.vx` - Velocity for trajectory prediction
/// 8. `ball.vy` - Velocity for bounce angle estimation
/// 
/// This representation satisfies the **Markov property**: the current state
/// contains all information needed for optimal decision-making.
pub const INPUT_SIZE: usize = LAYERS[0];

/// Number of neurons in the first hidden layer.
/// 
/// # Teaching Note: Hidden Layer Sizing
/// 16 neurons provide enough capacity for **feature detection** without overfitting.
/// This size follows the rule of thumb: hidden layer should be 2-3x the input size
/// for sufficient representational power while maintaining training efficiency.
pub const HIDDEN1_SIZE: usize = LAYERS[1];

/// Number of neurons in the second hidden layer.
///
/// # Teaching Note: Hierarchical Abstraction
/// 4 neurons force the network to **compress and abstract** the 16 features
/// into a smaller set of high-level decisions. This bottleneck architecture
/// encourages the network to learn meaningful, interpretable strategies.
pub const HIDDEN2_SIZE: usize = LAYERS[2];

/// Number of output neurons controlling paddle movement.
///
/// # Teaching Note: Output Design Philosophy
/// Single output simplifies the action space to 1D movement:
/// - Output > 0: Move paddle up
/// - Output < 0: Move paddle down  
/// - Output ≈ 0: Hold current position
/// 
/// This **continuous control** is more natural than discrete actions (up/down/stay)
/// and allows for nuanced velocity control and fine positioning.
pub const OUTPUT_SIZE: usize = LAYERS[3];

// ----------------------------------------------------------------------------
// Weight Calculation Constants (Neural Network Parameters)
// ----------------------------------------------------------------------------

/// Number of weights connecting input layer to first hidden layer.
///
/// # Teaching Note: Bias Neurons and Weight Calculation
/// The formula `HIDDEN1_SIZE * (INPUT_SIZE + 1)` accounts for **bias neurons**:
/// 
/// ## Without Bias: 
/// Each hidden neuron connects to 8 inputs = 8 weights per neuron
/// Total: 16 neurons × 8 weights = 128 weights
/// 
/// ## With Bias:
/// Each hidden neuron also connects to a bias (constant 1.0) = +1 weight per neuron
/// Total: 16 neurons × 9 weights = 144 weights
/// 
/// **Bias neurons** allow the activation function to shift left/right on the x-axis,
/// dramatically increasing the network's expressiveness. Without bias, neurons can
/// only learn patterns that pass through the origin.
pub const L1_WEIGHTS: usize = HIDDEN1_SIZE * (INPUT_SIZE + 1); // 16 * 9 = 144

/// Number of weights connecting first hidden layer to second hidden layer.
pub const L2_WEIGHTS: usize = HIDDEN2_SIZE * (HIDDEN1_SIZE + 1); // 4 * 17 = 68

/// Number of weights connecting second hidden layer to output layer.
pub const L3_WEIGHTS: usize = OUTPUT_SIZE * (HIDDEN2_SIZE + 1); // 1 * 5 = 5

/// The total number of weights in the neural network (the "genome" size).
///
/// # Teaching Note: Parameter Count Analysis
/// Total: 144 + 68 + 5 = **217 parameters**
/// 
/// This is the **genetic algorithm search space** - each individual's genome
/// contains 217 real numbers that must be optimized. For perspective:
/// - **Small enough**: Evolution can explore effectively (not too many dimensions)
/// - **Large enough**: Network can learn complex strategies (sufficient capacity)
/// - **Comparable**: Modern deep networks have millions/billions of parameters
/// 
/// ## Memory Usage:
/// - Per individual: 217 × 4 bytes = 868 bytes
/// - Population of 128: 217 × 128 × 4 = ~111 KB
/// - Very memory efficient for real-time evolution
pub const TOTAL_WEIGHTS: usize = L1_WEIGHTS + L2_WEIGHTS + L3_WEIGHTS; // 217 total 