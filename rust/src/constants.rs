//! Defines compile-time constants used throughout the application.
//!
//! This module centralizes all the magic numbers and configuration values that are
//! fixed at compile time. This improves maintainability, as critical values for the
//! simulation and neural network can be adjusted in one place.
//!
//! # Teaching Note: Constants vs Configuration
//! Constants defined here are **compile-time immutable** and represent fundamental
//! properties of the system (like physics laws or network architecture). They differ
//! from runtime configuration (in `config.rs`) which controls algorithm behavior.
//! This separation follows the principle of **compile-time vs runtime decisions**.
//!
//! # Architecture Philosophy
//! These constants define a **minimal viable Pong AI** - the simplest configuration
//! that can learn effective gameplay. Every choice here balances:
//! - **Computational efficiency** vs **representational power**
//! - **Learning speed** vs **final performance**
//! - **Implementation simplicity** vs **algorithmic sophistication**

// ----------------------------------------------------------------------------
// Game World Constants (Pong Environment)
// ----------------------------------------------------------------------------

/// The width of the Pong game area in pixels.
/// 
/// # Teaching Note: Coordinate System Design
/// This establishes the game's coordinate system. In Pong, the width is typically
/// wider than height to create the classic "tennis court" aspect ratio.
/// The 4:3 ratio (400:300) is chosen for:
/// - **Visual clarity**: Easy to see paddle and ball movements
/// - **Computational efficiency**: Nice round numbers for physics calculations
/// - **Historical accuracy**: Matches classic Pong proportions
pub const WIDTH: u16 = 400;

/// The height of the Pong game area in pixels.
///
/// # Teaching Note: Aspect Ratio Considerations
/// The height is intentionally smaller than width to create a landscape orientation.
/// This affects gameplay strategy - vertical movement is more constrained than
/// horizontal ball travel, making timing and positioning critical skills for AI.
pub const HEIGHT: u16 = 300;

/// The height of each paddle in pixels.
///
/// # Teaching Note: Game Balance Design
/// Paddle height (37.5px) is exactly 1/8th of the game height. This ratio determines
/// the difficulty of the game:
/// - **Too large**: Game becomes trivial, no challenge for AI
/// - **Too small**: Game becomes impossible, AI can't learn effectively
/// - **Just right**: Requires skill but allows successful defensive play
/// 
/// This value directly impacts the **fitness landscape** that the AI must learn to navigate.
pub const PADDLE_HEIGHT: f32 = HEIGHT as f32 / 8.0; // 37.5 pixels

/// The maximum score before a game ends.
///
/// # Teaching Note: Episode Length Design
/// Short games (max score = 1) create **dense learning signals**:
/// - Each point matters enormously for fitness evaluation
/// - Quick episode turnover allows more fitness evaluations per generation
/// - Reduces variance in fitness measurements
/// - Focuses learning on critical moments rather than endurance
pub const MAX_SCORE: u8 = 1;

// ----------------------------------------------------------------------------
// Physics and Simulation Constants
// ----------------------------------------------------------------------------

/// The number of simulation ticks that occur per second.
///
/// # Teaching Note: Temporal Discretization
/// This controls the simulation's temporal resolution. Higher values provide:
/// - **Better physics accuracy**: Smaller time steps reduce numerical integration errors
/// - **Smoother motion**: More position updates create fluid movement
/// - **Higher computational cost**: More calculations per simulated second
/// 
/// 60 FPS is chosen as the sweet spot for real-time feel without excessive computation.
pub const TICK_RATE: u16 = 60;

/// The maximum velocity of a paddle in pixels per tick.
///
/// # Teaching Note: Movement Constraints
/// This value (5.0 px/tick) determines how quickly paddles can react:
/// - **Too high**: Paddles become teleporters, game loses realism
/// - **Too low**: Paddles can't react to fast balls, game becomes impossible
/// - **Just right**: Requires prediction and positioning strategy
/// 
/// The formula `HEIGHT / TICK_RATE` ensures paddles can traverse the full height
/// in exactly 1 second, providing intuitive scaling.
pub const PADDLE_MAX_VEL: f32 = HEIGHT as f32 / TICK_RATE as f32; // 5.0 px/tick

/// The maximum number of steps (frames) a single game can last.
/// 
/// # Teaching Note: Infinite Loop Prevention
/// This acts as a safeguard against games that might never end due to
/// two perfectly matched, purely defensive AIs. A limit of 1800 steps (30 seconds
/// at 60fps) ensures every game terminates, allowing fitness evaluation to proceed.
pub const MAX_STEPS: u32 = 1800;

/// The initial x-velocity of the ball in pixels per tick.
///
/// # Teaching Note: Ball Speed Tuning
/// Ball velocity (6.67 px/tick) determines game pace and difficulty:
/// - Crosses the field in ~60 ticks (1 second) for human-readable gameplay
/// - Fast enough to challenge AI reaction time
/// - Slow enough to allow learning and strategic positioning
pub const BALL_INITIAL_VEL_X: f32 = WIDTH as f32 / TICK_RATE as f32; // ~6.67 px/tick

/// The initial y-velocity of the ball in pixels per tick.
///
/// # Teaching Note: Symmetric Physics
/// Y-velocity matches X-velocity to create symmetric ball movement.
/// This ensures no inherent bias toward horizontal vs vertical strategies.
pub const BALL_INITIAL_VEL_Y: f32 = WIDTH as f32 / TICK_RATE as f32; // ~6.67 px/tick



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


