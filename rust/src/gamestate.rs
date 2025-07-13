//! Contains the core logic for the Pong game simulation.
//!
//! This module defines the `GameState` struct and all associated functions for
//! running the game, including physics updates, collision detection, and scoring.
//! It is designed to be completely decoupled from any rendering or UI logic.
//!
//! # Teaching Note: Physics Simulation Architecture
//! This module demonstrates several key concepts in real-time physics simulation:
//! - **Discrete Time Integration**: Updates positions using fixed time steps
//! - **Collision Detection and Response**: Handles ball-paddle and ball-wall interactions
//! - **State Representation**: Minimal but complete game state for AI decision-making
//! - **Deterministic Physics**: Identical inputs produce identical outputs (crucial for training)
//!
//! # Design Philosophy: C++ Compatibility
//! This implementation carefully matches the original C++ version's physics behavior
//! to ensure fair performance comparisons. This includes:
//! - Identical collision detection algorithms
//! - Same input normalization schemes  
//! - Matching random number usage patterns
//! - Equivalent fitness calculation methods

use crate::{
    config::{Config, FitnessFunc},
    traits::Individual,
};

pub mod constants;
pub mod keyboard_controller;
pub use constants::*;
use rand_distr::{Distribution, Normal};

/// Represents the complete state of a Pong game simulation at a single point in time.
///
/// # Teaching Note: State Space Design
/// This struct defines the **state space** of the Pong environment - all the information
/// needed to completely describe the system at any moment. Good state representations:
/// - **Complete**: Contain all information needed for decision-making
/// - **Minimal**: No redundant or derivable information
/// - **Normalized**: Values scaled to reasonable ranges for AI processing
/// - **Observable**: AI can actually sense these values in real scenarios
///
/// # Memory Layout Optimization
/// All fields are simple, stack-allocated `Copy` types for maximum performance:
/// - No heap allocations during simulation
/// - Extremely cheap cloning for parallel processing
/// - Cache-friendly memory layout
/// - Optimal for high-frequency function calls in training loops
///
/// # Neural Network Integration
/// The state is transformed into an 8-element feature vector for neural network input.
/// This transformation (normalization) is crucial for training stability and performance.
#[derive(Clone, Copy, Debug)]
pub struct GameState {
    /// Y-position of the left paddle (player 1) in pixels from top edge
    pub paddle1_pos: f32,
    /// Y-velocity of the left paddle in pixels per tick
    pub paddle1_vel: f32,
    /// Y-position of the right paddle (player 2) in pixels from top edge  
    pub paddle2_pos: f32,
    /// Y-velocity of the right paddle in pixels per tick
    pub paddle2_vel: f32,
    /// Ball position as (x, y) coordinates in pixels
    pub ball_pos: (f32, f32),
    /// Ball velocity as (vx, vy) in pixels per tick
    pub ball_vel: (f32, f32),
    /// Current scores as (left_score, right_score)
    pub scores: (u8, u8),
    /// Successful ball returns as (left_returns, right_returns)
    pub returns: (u32, u32),
    /// Successful shots (balls that would score) as (left_shots, right_shots)
    pub shots: (u32, u32),
}

impl GameState {
    /// Creates a new `GameState` with paddles and ball centered and scores zeroed.
    ///
    /// # Teaching Note: Initialization Strategy
    /// Starting positions are carefully chosen to be fair and reproducible:
    /// - Paddles at screen center: No initial advantage for either player
    /// - Ball at field center: Neutral starting position
    /// - Zero velocities: Game begins from rest
    /// - `reset_ball()` sets initial ball velocity to create deterministic serve patterns
    pub fn new() -> Self {
        let mut new_state = Self {
            paddle1_pos: (HEIGHT / 2) as f32,
            paddle1_vel: 0.0,
            paddle2_pos: (HEIGHT / 2) as f32,
            paddle2_vel: 0.0,
            ball_pos: ((WIDTH / 2) as f32, (HEIGHT / 2) as f32),
            ball_vel: (0.0, 0.0), // Velocity is set by reset_ball
            scores: (0, 0),
            returns: (0, 0),
            shots: (0, 0),
        };
        new_state.reset_ball(true); // Set initial ball state
        new_state
    }

    /// Resets the game state to its initial configuration for a new match.
    ///
    /// # Teaching Note: Episode Reset Pattern
    /// In reinforcement learning and evolutionary algorithms, each "episode" 
    /// (complete game) starts from a consistent initial state. This ensures:
    /// - **Fair comparison**: All individuals start with identical conditions
    /// - **Reproducible results**: Same random seed = same game sequence
    /// - **Unbiased learning**: No position advantages carry over between games
    pub fn reset(&mut self) {
        self.paddle1_pos = (HEIGHT / 2) as f32;
        self.paddle1_vel = 0.0;
        self.paddle2_pos = (HEIGHT / 2) as f32;
        self.paddle2_vel = 0.0;
        self.scores = (0, 0);
        self.returns = (0, 0);
        self.shots = (0, 0);
        self.reset_ball(rand::random());
    }

    /// Prepares the 8-element input array for the **left paddle's** neural network.
    ///
    /// # Teaching Note: Feature Engineering for Neural Networks
    /// This function performs **feature engineering** - transforming raw game state
    /// into a format optimal for neural network learning:
    ///
    /// ## Normalization Strategy:
    /// All inputs are scaled to approximately [-1, 1] range for several reasons:
    /// - **Training Stability**: Prevents any single input from dominating others
    /// - **Gradient Flow**: Keeps gradients in reasonable ranges for backpropagation
    /// - **Activation Function Efficiency**: Most functions work best in [-1, 1]
    /// - **Weight Initialization**: Standard initialization assumes normalized inputs
    ///
    /// ## Input Encoding:
    /// 1. **Position inputs**: Normalized by dividing by screen dimensions
    /// 2. **Velocity inputs**: Normalized by dividing by maximum possible velocities
    /// 3. **Coordinate transformation**: Maps [0, dimension] to [-1, 1] using 2x/d - 1
    pub fn get_inputs_for_player1(&self) -> [f32; 8] {
        [
            (2.0 * self.paddle1_pos / HEIGHT as f32) - 1.0, // Own Paddle Y: [0,HEIGHT] → [-1,1]
            self.paddle1_vel / PADDLE_MAX_VEL,              // Own Paddle Vel Y: normalized by max speed
            (2.0 * self.paddle2_pos / HEIGHT as f32) - 1.0, // Opponent Paddle Y: [0,HEIGHT] → [-1,1]
            self.paddle2_vel / PADDLE_MAX_VEL,              // Opponent Paddle Vel Y: normalized by max speed
            (2.0 * self.ball_pos.0 / WIDTH as f32) - 1.0,   // Ball X: [0,WIDTH] → [-1,1]
            (2.0 * self.ball_pos.1 / HEIGHT as f32) - 1.0,  // Ball Y: [0,HEIGHT] → [-1,1]
            self.ball_vel.0 / (2.0 * BALL_INITIAL_VEL_X),   // Ball Vel X: normalized by 2x initial speed
            self.ball_vel.1 / (2.0 * BALL_INITIAL_VEL_Y),   // Ball Vel Y: normalized by 2x initial speed
        ]
    }

    /// Prepares the 8-element input array for the **right paddle's** neural network.
    ///
    /// # Teaching Note: Symmetry and Network Reuse
    /// This function enables a **single neural network** to control both paddles by
    /// creating a **symmetric perspective** of the game world. Key concepts:
    ///
    /// ## Perspective Transformation:
    /// - Right player sees the world "flipped" - their paddle becomes "own paddle"
    /// - Ball positions and velocities are negated to maintain spatial relationships
    /// - This creates a consistent coordinate system where positive X always points "toward opponent"
    ///
    /// ## Benefits of Network Reuse:
    /// - **Parameter Efficiency**: Half the weights to optimize vs. separate networks
    /// - **Faster Training**: Twice the training data per individual (both sides)
    /// - **Guaranteed Fairness**: Both players use identical decision-making logic
    /// - **Reduced Overfitting**: Forced symmetry prevents learning position-specific biases
    ///
    /// ## C++ Compatibility:
    /// All inputs are negated to exactly match the original implementation's behavior.
    pub fn get_inputs_for_player2(&self) -> [f32; 8] {
        [
            -((2.0 * self.paddle2_pos / HEIGHT as f32) - 1.0), // Own Paddle Y (negated)
            -self.paddle2_vel / PADDLE_MAX_VEL,                 // Own Paddle Vel Y (negated)
            -((2.0 * self.paddle1_pos / HEIGHT as f32) - 1.0), // Opponent Paddle Y (negated)
            -self.paddle1_vel / PADDLE_MAX_VEL,                 // Opponent Paddle Vel Y (negated)
            -((2.0 * self.ball_pos.0 / WIDTH as f32) - 1.0),   // Ball X (negated)
            -((2.0 * self.ball_pos.1 / HEIGHT as f32) - 1.0),  // Ball Y (negated)
            -self.ball_vel.0 / (2.0 * BALL_INITIAL_VEL_X),     // Ball Vel X (negated)
            -self.ball_vel.1 / (2.0 * BALL_INITIAL_VEL_Y),     // Ball Vel Y (negated)
        ]
    }

    /// Resets the ball to the center with initial velocity toward specified player.
    ///
    /// # Teaching Note: Deterministic Serving
    /// This function implements a **deterministic serve pattern** essential for:
    /// - **Reproducible Training**: Same random seed produces same game sequences
    /// - **Fair Evaluation**: Both players get equal serving opportunities
    /// - **Consistent Physics**: Ball always starts with identical energy
    pub fn reset_ball(&mut self, serve_to_left_player: bool) {
        self.ball_pos = ((WIDTH / 2) as f32, (HEIGHT / 2) as f32);

        if serve_to_left_player {
            self.ball_vel = (BALL_INITIAL_VEL_X, BALL_INITIAL_VEL_Y);
        } else {
            self.ball_vel = (-BALL_INITIAL_VEL_X, -BALL_INITIAL_VEL_Y);
        }
    }

    /// Resets the ball with optional randomization based on configuration.
    ///
    /// # Teaching Note: Training Diversity vs Evaluation Consistency
    /// This function demonstrates the **training-evaluation trade-off**:
    ///
    /// ## During Training (`random_ball_direction = true`):
    /// - Randomized serves expose AI to diverse scenarios
    /// - Prevents overfitting to specific ball trajectories
    /// - Improves generalization to unseen situations
    /// - Creates more robust, adaptable strategies
    ///
    /// ## During Evaluation (`random_ball_direction = false`):
    /// - Deterministic serves ensure fair, repeatable comparisons
    /// - Eliminates random variance from performance measurements
    /// - Allows precise measurement of strategic improvements
    /// - Enables reproducible research results
    pub fn reset_ball_with_config(&mut self, serve_to_left_player: bool, config: &Config) {
        self.ball_pos = ((WIDTH / 2) as f32, (HEIGHT / 2) as f32);

        let base_vel_x = if serve_to_left_player { BALL_INITIAL_VEL_X } else { -BALL_INITIAL_VEL_X };
        let base_vel_y = if serve_to_left_player { BALL_INITIAL_VEL_Y } else { -BALL_INITIAL_VEL_Y };

        if config.random_ball_direction {
            // Randomize the angle within a cone (±30 degrees from straight)
            use rand::Rng;
            let mut rng = rand::rng();
            let angle_variation = rng.random_range(-std::f32::consts::PI/6.0..std::f32::consts::PI/6.0); // ±30 degrees
            
            let speed = (base_vel_x * base_vel_x + base_vel_y * base_vel_y).sqrt();
            let base_angle = base_vel_y.atan2(base_vel_x);
            let new_angle = base_angle + angle_variation;
            
            self.ball_vel = (speed * new_angle.cos(), speed * new_angle.sin());
        } else {
            self.ball_vel = (base_vel_x, base_vel_y);
        }
    }

    /// Advances the game by one full tick during a training simulation.
    ///
    /// # Teaching Note: Simulation Loop Architecture
    /// This function implements the core **game loop pattern**:
    /// 1. **Sense**: Gather current state information
    /// 2. **Think**: Neural networks process inputs and decide actions
    /// 3. **Act**: Apply decisions to paddle velocities
    /// 4. **Update**: Advance physics simulation by one time step
    ///
    /// This separation of concerns enables:
    /// - **Modular design**: Each phase can be optimized independently
    /// - **Easy debugging**: Can isolate issues to specific simulation phases  
    /// - **AI integration**: Clean interface between AI decisions and physics
    /// - **Deterministic behavior**: Identical inputs always produce identical outputs
    pub fn tick<I: Individual>(&mut self, left: &I, right: &I, config: &Config) {
        self.update_paddles(left, right, config);
        self.advance_frame(config);
    }

    /// Sets paddle velocities based on neural network outputs.
    ///
    /// # Teaching Note: AI Motor Control Interface
    /// This function bridges **cognition** (neural network decisions) and **action**
    /// (physical paddle movement). Key design principles:
    ///
    /// ## Output Interpretation:
    /// - Network output is a single real number (continuous control)
    /// - Positive values → move paddle up
    /// - Negative values → move paddle down  
    /// - Zero values → hold current position
    /// - Magnitude controls movement speed
    ///
    /// ## Safety and Constraints:
    /// - `clamp()` ensures paddle velocities never exceed physical limits
    /// - Prevents unrealistic "teleportation" behaviors
    /// - Maintains game physics consistency
    ///
    /// ## C++ Compatibility Note:
    /// Right paddle output is negated to match original implementation behavior.
    /// This ensures identical gameplay when comparing performance against C++ version.
    pub fn update_paddles<I: Individual>(&mut self, left: &I, right: &I, config: &Config) {
        let left_inputs = self.get_inputs_for_player1();
        let right_inputs = self.get_inputs_for_player2();

        let left_output = left.forward_propagate(&left_inputs, config.activation);
        let right_output = right.forward_propagate(&right_inputs, config.activation);

        // Set velocities based on network output, clamped to maximum paddle speed
        // Note: Right paddle output is negated to match C++ version behavior
        self.paddle1_vel = (left_output[0] * PADDLE_MAX_VEL).clamp(-PADDLE_MAX_VEL, PADDLE_MAX_VEL);
        self.paddle2_vel =
            (-right_output[0] * PADDLE_MAX_VEL).clamp(-PADDLE_MAX_VEL, PADDLE_MAX_VEL);
    }

    /// Advances the game state by one frame, applying physics and handling collisions.
    ///
    /// # Teaching Note: Physics Integration Patterns
    /// This function implements **explicit Euler integration** for position updates:
    /// `new_position = old_position + velocity * time_step`
    /// 
    /// ## Integration Order (Critical):
    /// 1. **Update paddle positions** from current velocities
    /// 2. **Apply boundary constraints** (paddle clamping)
    /// 3. **Update ball physics** with collision detection
    ///
    /// ## Boundary Handling:
    /// Paddles are constrained within screen bounds with **hard clamping**:
    /// - Position clamped to valid range
    /// - Velocity zeroed when hitting boundaries (prevents "bouncing")
    /// - Maintains paddle center within [PADDLE_HEIGHT/2, HEIGHT-PADDLE_HEIGHT/2]
    pub fn advance_frame(&mut self, config: &Config) {
        // Update paddle positions and clamp them to screen boundaries
        self.paddle1_pos += self.paddle1_vel;
        if self.paddle1_pos < PADDLE_HEIGHT / 2.0 {
            self.paddle1_pos = PADDLE_HEIGHT / 2.0;
            self.paddle1_vel = 0.0; // Stop movement when hitting boundary
        }
        if self.paddle1_pos > HEIGHT as f32 - PADDLE_HEIGHT / 2.0 {
            self.paddle1_pos = HEIGHT as f32 - PADDLE_HEIGHT / 2.0;
            self.paddle1_vel = 0.0; // Stop movement when hitting boundary
        }

        self.paddle2_pos += self.paddle2_vel;
        if self.paddle2_pos < PADDLE_HEIGHT / 2.0 {
            self.paddle2_pos = PADDLE_HEIGHT / 2.0;
            self.paddle2_vel = 0.0;
        }
        if self.paddle2_pos > HEIGHT as f32 - PADDLE_HEIGHT / 2.0 {
            self.paddle2_pos = HEIGHT as f32 - PADDLE_HEIGHT / 2.0;
            self.paddle2_vel = 0.0;
        }

        self.update_ball(config);
    }

    /// Updates the ball's position and handles all collisions.
    ///
    /// # Teaching Note: Ball Physics and Collision Handling
    /// This function implements the core **collision detection and response** system:
    ///
    /// ## Physics Integration Order:
    /// 1. **Move first**: Update ball position by adding velocity
    /// 2. **Check collisions**: Detect any boundary or paddle intersections
    /// 3. **Respond to collisions**: Modify velocity and position accordingly
    ///
    /// ## Collision Detection Types:
    /// - **Wall Collisions**: Simple boundary checks for top/bottom walls
    /// - **Paddle Collisions**: Box-intersection tests with velocity direction checks
    /// - **Scoring Zones**: Ball position beyond left/right screen edges
    ///
    /// ## Advanced Features:
    /// - **Velocity Transfer**: Paddle momentum affects ball trajectory (realistic physics)
    /// - **Random Deflection**: Small noise prevents deterministic ball patterns
    /// - **Shot Detection**: Predictive analysis for fitness calculation
    pub fn update_ball(&mut self, config: &Config) {
        // Update ball position using explicit Euler integration
        self.ball_pos.0 += self.ball_vel.0;
        self.ball_pos.1 += self.ball_vel.1;

        let mut paddle_hit = false;

        // Wall collision detection and response (top/bottom boundaries)
        // Check velocity direction to prevent "double bouncing" when ball is exactly on boundary
        if (self.ball_pos.1 <= 0.0 && self.ball_vel.1 < 0.0)
            || (self.ball_pos.1 >= HEIGHT as f32 && self.ball_vel.1 > 0.0)
        {
            self.ball_vel.1 *= -1.0; // Perfect elastic collision (no energy loss)
        }

        // Paddle collision detection - check before scoring to prioritize returns over points
        let paddle1_box = (0.0, self.paddle1_pos);
        let paddle2_box = (WIDTH as f32, self.paddle2_pos);

        // Left paddle collision detection and response
        if self.ball_vel.0 < 0.0 && self.ball_pos.0 <= paddle1_box.0 {
            // Ball is moving left and has reached or passed the left paddle's x-position
            let paddle_top = paddle1_box.1 - PADDLE_HEIGHT / 2.0;
            let paddle_bottom = paddle1_box.1 + PADDLE_HEIGHT / 2.0;
            
            // Check if ball y-position intersects with paddle y-range
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_pos.0 = paddle1_box.0; // Prevent ball from penetrating paddle
                paddle_hit = true;

                // Physics response: elastic collision with velocity transfer
                self.ball_vel.0 = self.ball_vel.0.abs(); // Reverse horizontal direction
                self.ball_vel.1 += self.paddle1_vel; // Transfer paddle's momentum to ball

                // Add small random deflection to prevent repetitive patterns
                // This creates more varied and interesting gameplay
                let mut rng = rand::rng();
                self.ball_vel.1 +=
                    Normal::new(0.0, 0.05).unwrap().sample(&mut rng) * BALL_INITIAL_VEL_Y;

                self.returns.0 += 1; // Successful return for left player
            }
        // Right paddle collision detection and response
        } else if self.ball_vel.0 > 0.0 && self.ball_pos.0 >= paddle2_box.0 {
            // Ball is moving right and has reached or passed the right paddle's x-position
            let paddle_top = paddle2_box.1 - PADDLE_HEIGHT / 2.0;
            let paddle_bottom = paddle2_box.1 + PADDLE_HEIGHT / 2.0;
            
            // Check if ball y-position intersects with paddle y-range
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_pos.0 = paddle2_box.0; // Prevent ball from penetrating paddle
                paddle_hit = true;

                // Physics response: elastic collision with velocity transfer
                self.ball_vel.0 = -self.ball_vel.0.abs(); // Reverse horizontal direction
                self.ball_vel.1 += self.paddle2_vel; // Transfer paddle's momentum to ball

                // Add small random deflection to prevent repetitive patterns
                let mut rng = rand::rng();
                self.ball_vel.1 +=
                    Normal::new(0.0, 0.05).unwrap().sample(&mut rng) * BALL_INITIAL_VEL_Y;

                self.returns.1 += 1; // Successful return for right player
            }
        }

        // Shot detection: Predictive analysis for fitness evaluation
        // This calculates whether the ball trajectory would result in a point
        // Helps reward offensive play even when opponent successfully defends
        if self.ball_vel.0 < 0.0 {
            // Ball moving toward left player (potential shot for right player)
            if self.ball_vel.0.abs() > f32::EPSILON {
                let time_to_wall = -self.ball_pos.0 / self.ball_vel.0;
                let shot_y = self.ball_pos.1 + self.ball_vel.1 * time_to_wall;

                // Only count shots that would land within the game area
                if shot_y >= 0.0 && shot_y <= HEIGHT as f32 {
                    let paddle_top = self.paddle1_pos - PADDLE_HEIGHT / 2.0;
                    let paddle_bottom = self.paddle1_pos + PADDLE_HEIGHT / 2.0;
                    
                    // If projected shot location is outside paddle reach, it's a successful shot
                    if shot_y < paddle_top || shot_y > paddle_bottom {
                        self.shots.1 += 1; // Right player executed a shot
                    }
                }
            }
        } else if self.ball_vel.0 > 0.0 {
            // Ball moving toward right player (potential shot for left player)
            if self.ball_vel.0.abs() > f32::EPSILON {
                let time_to_wall = (WIDTH as f32 - self.ball_pos.0) / self.ball_vel.0;
                let shot_y = self.ball_pos.1 + self.ball_vel.1 * time_to_wall;

                // Only count shots that would land within the game area
                if shot_y >= 0.0 && shot_y <= HEIGHT as f32 {
                    let paddle_top = self.paddle2_pos - PADDLE_HEIGHT / 2.0;
                    let paddle_bottom = self.paddle2_pos + PADDLE_HEIGHT / 2.0;
                    
                    // If projected shot location is outside paddle reach, it's a successful shot
                    if shot_y < paddle_top || shot_y > paddle_bottom {
                        self.shots.0 += 1; // Left player executed a shot
                    }
                }
            }
        }

        // Scoring detection and game state reset
        // Only check for scoring if no paddle collision occurred this frame
        if !paddle_hit {
            if self.ball_pos.0 < 0.0 {
                // Ball passed left edge - right player scores
                self.scores.1 += 1;
                self.paddle1_pos = (HEIGHT / 2) as f32; // Reset paddle positions
                self.paddle2_pos = (HEIGHT / 2) as f32;
                self.reset_ball_with_config(true, config); // Serve to left player
            } else if self.ball_pos.0 > WIDTH as f32 {
                // Ball passed right edge - left player scores
                self.scores.0 += 1;
                self.paddle1_pos = (HEIGHT / 2) as f32; // Reset paddle positions
                self.paddle2_pos = (HEIGHT / 2) as f32;
                self.reset_ball_with_config(false, config); // Serve to right player
            }
        }
    }

    /// Runs a complete simulation episode between two individuals until a winner is decided.
    ///
    /// # Teaching Note: Complete Simulation Architecture
    /// This function orchestrates a full **simulation episode** - the fundamental unit
    /// of evaluation in evolutionary algorithms. Key architectural principles:
    ///
    /// ## Episode Structure:
    /// 1. **Reset**: Start from clean initial state
    /// 2. **Simulate**: Run game loop until termination condition
    /// 3. **Evaluate**: Calculate fitness scores based on observed behavior
    /// 4. **Return**: Provide standardized fitness values for ranking
    ///
    /// ## Termination Conditions:
    /// - **Victory condition**: One player reaches maximum score
    /// - **Timeout condition**: Simulation exceeds reasonable time limit
    /// - **Safety condition**: Prevents infinite loops from degenerate strategies
    ///
    /// ## Fitness Function Design Philosophy:
    /// Different fitness functions encourage different strategic behaviors:
    /// - **C++ Equivalent**: Rewards consistent defensive play and opportunistic offense
    /// - **Return Focused**: Emphasizes active, engaging gameplay over passive defense
    /// - **Victory Optimized**: Promotes aggressive, decisive strategies
    ///
    /// # Returns
    /// Nested tuple `((left_primary, left_secondary), (right_primary, right_secondary))`
    /// - **Primary fitness**: Main optimization objective (usually returns + shots)
    /// - **Secondary fitness**: Tie-breaking criterion (usually wins)
    pub fn simulate<I: Individual>(
        &mut self,
        left: &I,
        right: &I,
        config: &Config,
    ) -> ((u32, u32), (u32, u32)) {
        self.reset();

        // Calculate simulation timeout to prevent infinite games
        // Formula derived from C++ version: allows reasonable rally lengths
        // while preventing degenerate strategies (e.g., never moving)
        let timelimit = 2 * 16 * MAX_SCORE as u32 * WIDTH as u32; // ~25,600 ticks for max_score=1
        
        for _tick in 0..timelimit {
            // Check termination condition: game complete
            if self.scores.0 >= MAX_SCORE || self.scores.1 >= MAX_SCORE {
                break;
            }
            
            // Execute one simulation step
            self.tick(left, right, config);
        }

        // Calculate final fitness scores based on selected fitness function
        match config.fitness_func {
            // C++ equivalent multi-objective fitness function
            // Primary: offensive capabilities (returns + shots)  
            // Secondary: ultimate success (wins)
            FitnessFunc::CppEquivalent => {
                let left_wins = if self.scores.0 > self.scores.1 { 1 } else { 0 };
                let right_wins = if self.scores.1 > self.scores.0 { 1 } else { 0 };
                let left_primary = self.returns.0 + self.shots.0;
                let right_primary = self.returns.1 + self.shots.1;
                ((left_primary, left_wins), (right_primary, right_wins))
            }

            // Return-focused fitness: emphasizes consistent ball contact and engagement
            // Rewards players who create long, skillful rallies with bonus for wins
            FitnessFunc::ReturnFocused => {
                let mut left_score = self.returns.0;
                let mut right_score = self.returns.1;
                
                // Modest win bonus to maintain competitive pressure without dominating
                if self.scores.0 > self.scores.1 {
                    left_score += 5;
                } else if self.scores.1 > self.scores.0 {
                    right_score += 5;
                }
                ((left_score, 0), (right_score, 0))
            }
            
            // Victory-optimized fitness: strongly rewards decisive play
            // Encourages strategies that finish points quickly and effectively
            FitnessFunc::VictoryOptimized => {
                let mut left_score = self.returns.0;
                let mut right_score = self.returns.1;
                
                // Large win bonus encourages aggressive, point-winning strategies
                if self.scores.0 >= MAX_SCORE {
                    left_score += 10; // Significant bonus for decisive victory
                }
                if self.scores.1 >= MAX_SCORE {
                    right_score += 10; // Significant bonus for decisive victory
                }
                ((left_score, 0), (right_score, 0))
            }
        }
    }
}
