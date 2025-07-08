//! Contains the core logic for the Pong game simulation.
//!
//! This module defines the `GameState` struct and all associated functions for
//! running the game, including physics updates, collision detection, and scoring.
//! It is designed to be completely decoupled from any rendering or UI logic.

use crate::{
    config::{Config, FitnessFunc},
    constants::*,
    traits::Individual,
};
use rand_distr::{Distribution, Normal};

/// Represents the full state of a Pong game simulation at a single point in time.
///
/// # Memory Layout
/// All fields are simple, stack-allocated `Copy` types. This makes the entire
/// `GameState` struct `Copy`-able, allowing for extremely cheap cloning. This is
/// beneficial in genetic algorithms where many simulations might be run in parallel
/// from a common starting state.
///
/// # Neural Network Input
/// The game state is converted into an 8-element array before being fed to the
/// neural network. See the documentation for `constants::INPUT_SIZE` for a detailed
/// breakdown of these inputs. All inputs are normalized to a consistent range
/// (e.g., `[-1, 1]`) to improve network training stability.
///
/// # Teaching Note
/// This struct is a prime example of a **state representation**. It captures the
/// minimum amount of information needed to fully describe the system at any instant.
/// The choice of what to include is critical for both the simulation's correctness
/// and the neural network's ability to make informed decisions.
#[derive(Clone, Copy, Debug)]
pub struct GameState {
    pub paddle1_pos: f32,
    pub paddle1_vel: f32,
    pub paddle2_pos: f32,
    pub paddle2_vel: f32,
    pub ball_pos: (f32, f32),
    pub ball_vel: (f32, f32),
    pub scores: (u8, u8),
    pub returns: (u32, u32),
    pub shots: (u32, u32),
}

impl GameState {
    /// Creates a new `GameState` with paddles and ball centered and scores zeroed.
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
    /// # Teaching Note
    /// Normalization (dividing by constants like `WIDTH` or `PADDLE_MAX_VEL`) is a crucial
    /// preprocessing step. It scales all inputs to a similar range (usually `[-1, 1]` or
    /// `[0, 1]`), which helps the neural network train more effectively and prevents certain
    /// inputs from disproportionately influencing the outcome.
    pub fn get_inputs_for_player1(&self) -> [f32; 8] {
        [
            (2.0 * self.paddle1_pos / HEIGHT as f32) - 1.0, // Own Paddle Y
            self.paddle1_vel / PADDLE_MAX_VEL,              // Own Paddle Vel Y
            (2.0 * self.paddle2_pos / HEIGHT as f32) - 1.0, // Opponent Paddle Y
            self.paddle2_vel / PADDLE_MAX_VEL,              // Opponent Paddle Vel Y
            (2.0 * self.ball_pos.0 / WIDTH as f32) - 1.0,   // Ball X
            (2.0 * self.ball_pos.1 / HEIGHT as f32) - 1.0,  // Ball Y
            self.ball_vel.0 / (2.0 * BALL_INITIAL_VEL_X),   // Ball Vel X
            self.ball_vel.1 / (2.0 * BALL_INITIAL_VEL_Y),   // Ball Vel Y
        ]
    }

    /// Prepares the 8-element input array for the **right paddle's** neural network.
    ///
    /// # Teaching Note
    /// To allow a single neural network to control both paddles, we can reuse the same
    /// network by simply flipping its perspective of the world. This matches the C++ version
    /// exactly where ALL inputs are negated for the right player.
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

    /// Resets the ball to the center with velocity, optionally randomized.
    pub fn reset_ball(&mut self, serve_to_left_player: bool) {
        self.ball_pos = ((WIDTH / 2) as f32, (HEIGHT / 2) as f32);

        if serve_to_left_player {
            self.ball_vel = (BALL_INITIAL_VEL_X, BALL_INITIAL_VEL_Y);
        } else {
            self.ball_vel = (-BALL_INITIAL_VEL_X, -BALL_INITIAL_VEL_Y);
        }
    }

    /// Resets the ball with optional randomization based on config.
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
    /// # Algorithm
    /// 1.  `update_paddles`: Get decisions from the neural networks and set paddle velocities.
    /// 2.  `advance_frame`: Update all positions and handle physics/collisions for the tick.
    pub fn tick<I: Individual>(&mut self, left: &I, right: &I, config: &Config) {
        self.update_paddles(left, right, config);
        self.advance_frame(config);
    }

    /// Sets paddle velocities based on neural network outputs.
    ///
    /// # Teaching Note
    /// This function is the bridge between the AI's 'thought' (the network output) and its
    /// 'action' (the paddle's velocity). The output is a single float, which we scale and
    /// clamp to a valid range. Note that this function *only* sets the velocity; the actual
    /// movement happens in `advance_frame`.
    pub fn update_paddles<I: Individual>(&mut self, left: &I, right: &I, config: &Config) {
        let left_inputs = self.get_inputs_for_player1();
        let right_inputs = self.get_inputs_for_player2();

        let left_output = left.forward_propagate(&left_inputs, config.activation);
        let right_output = right.forward_propagate(&right_inputs, config.activation);

        // Set velocities based on net output, clamped to the maximum paddle speed.
        // Note: Right paddle output is negated to match C++ version behavior
        self.paddle1_vel = (left_output[0] * PADDLE_MAX_VEL).clamp(-PADDLE_MAX_VEL, PADDLE_MAX_VEL);
        self.paddle2_vel =
            (-right_output[0] * PADDLE_MAX_VEL).clamp(-PADDLE_MAX_VEL, PADDLE_MAX_VEL);
    }

    /// Advances the game state by one frame, applying physics and handling collisions.
    ///
    /// This function is used by both the training simulation (`tick`) and the interactive TUI.
    /// It assumes paddle velocities have already been set for the current frame.
    ///
    /// # Algorithm
    /// 1.  Update paddle positions based on their current velocities, clamping to screen bounds.
    /// 2.  Call `update_ball` to handle all ball-related physics for the frame.
    pub fn advance_frame(&mut self, config: &Config) {
        // Update paddle positions and clamp them to the screen boundaries.
        self.paddle1_pos += self.paddle1_vel;
        if self.paddle1_pos < PADDLE_HEIGHT / 2.0 {
            self.paddle1_pos = PADDLE_HEIGHT / 2.0;
            self.paddle1_vel = 0.0;
        }
        if self.paddle1_pos > HEIGHT as f32 - PADDLE_HEIGHT / 2.0 {
            self.paddle1_pos = HEIGHT as f32 - PADDLE_HEIGHT / 2.0;
            self.paddle1_vel = 0.0;
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
    /// # Teaching Note
    /// This function contains the core physics logic. The order of operations is important:
    /// move, then check for collisions. Collision checks use the ball's center position for accuracy.
    /// The paddle collision logic includes imparting some of the paddle's velocity to the ball,
    /// creating more dynamic and interesting rallies.
    pub fn update_ball(&mut self, config: &Config) {
        self.ball_pos.0 += self.ball_vel.0;
        self.ball_pos.1 += self.ball_vel.1;

        let mut paddle_hit = false;

        // Top/Bottom wall collision
        if (self.ball_pos.1 <= 0.0 && self.ball_vel.1 < 0.0)
            || (self.ball_pos.1 >= HEIGHT as f32 && self.ball_vel.1 > 0.0)
        {
            self.ball_vel.1 *= -1.0;
        }

        // Paddle collision detection - check before scoring
        let paddle1_box = (0.0, self.paddle1_pos);
        let paddle2_box = (WIDTH as f32, self.paddle2_pos);

        // Left paddle collision
        if self.ball_vel.0 < 0.0 && self.ball_pos.0 <= paddle1_box.0 {
            let paddle_top = paddle1_box.1 - PADDLE_HEIGHT / 2.0;
            let paddle_bottom = paddle1_box.1 + PADDLE_HEIGHT / 2.0;
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_pos.0 = paddle1_box.0; // Clamp ball to paddle front
                paddle_hit = true;

                self.ball_vel.0 = self.ball_vel.0.abs(); // Reflect ball horizontally
                self.ball_vel.1 += self.paddle1_vel; // Add paddle's velocity

                // Add random deflection
                let mut rng = rand::rng();
                self.ball_vel.1 +=
                    Normal::new(0.0, 0.05).unwrap().sample(&mut rng) * BALL_INITIAL_VEL_Y;

                self.returns.0 += 1;
            }
        // Right paddle collision
        } else if self.ball_vel.0 > 0.0 && self.ball_pos.0 >= paddle2_box.0 {
            let paddle_top = paddle2_box.1 - PADDLE_HEIGHT / 2.0;
            let paddle_bottom = paddle2_box.1 + PADDLE_HEIGHT / 2.0;
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_pos.0 = paddle2_box.0; // Clamp ball to paddle front
                paddle_hit = true;

                self.ball_vel.0 = -self.ball_vel.0.abs(); // Reflect ball horizontally
                self.ball_vel.1 += self.paddle2_vel; // Add paddle's velocity

                // Add random deflection
                let mut rng = rand::rng();
                self.ball_vel.1 +=
                    Normal::new(0.0, 0.05).unwrap().sample(&mut rng) * BALL_INITIAL_VEL_Y;

                self.returns.1 += 1;
            }
        }

        // A shot is a tick in which the ball is headed to score.
        // This logic projects the ball's trajectory to the paddle line and checks if the
        // paddle can intercept it. This is derived from the original C++ implementation.
        if self.ball_vel.0 < 0.0 {
            // Ball moving left, potential shot for player 2 (right)
            // Project ball's path to the left wall (x=0)
            if self.ball_vel.0.abs() > f32::EPSILON {
                let time_to_wall = -self.ball_pos.0 / self.ball_vel.0;
                let shot_y = self.ball_pos.1 + self.ball_vel.1 * time_to_wall;

                // Check if the projected position is on the screen
                if shot_y >= 0.0 && shot_y <= HEIGHT as f32 {
                    let paddle_top = self.paddle1_pos - PADDLE_HEIGHT / 2.0;
                    let paddle_bottom = self.paddle1_pos + PADDLE_HEIGHT / 2.0;
                    // If the projected position is outside the paddle's reach, it's a successful shot
                    if shot_y < paddle_top || shot_y > paddle_bottom {
                        self.shots.1 += 1; // right player's shot
                    }
                }
            }
        } else if self.ball_vel.0 > 0.0 {
            // Ball moving right, potential shot for player 1 (left)
            // Project ball's path to the right wall (x=WIDTH)
            if self.ball_vel.0.abs() > f32::EPSILON {
                let time_to_wall = (WIDTH as f32 - self.ball_pos.0) / self.ball_vel.0;
                let shot_y = self.ball_pos.1 + self.ball_vel.1 * time_to_wall;

                // Check if the projected position is on the screen
                if shot_y >= 0.0 && shot_y <= HEIGHT as f32 {
                    let paddle_top = self.paddle2_pos - PADDLE_HEIGHT / 2.0;
                    let paddle_bottom = self.paddle2_pos + PADDLE_HEIGHT / 2.0;
                    // If the projected position is outside the paddle's reach, it's a successful shot
                    if shot_y < paddle_top || shot_y > paddle_bottom {
                        self.shots.0 += 1; // left player's shot
                    }
                }
            }
        }

        // Score detection. Check if the ball has passed the screen edge.
        // This matches the C++ version which uses ball center position.
        if !paddle_hit {
            if self.ball_pos.0 < 0.0 {
                self.scores.1 += 1; // Player 2 (right) scores
                self.paddle1_pos = (HEIGHT / 2) as f32;
                self.paddle2_pos = (HEIGHT / 2) as f32;
                self.reset_ball_with_config(true, config); // Serve to player 2 (right)
            } else if self.ball_pos.0 > WIDTH as f32 {
                self.scores.0 += 1; // Player 1 (left) scores
                self.paddle1_pos = (HEIGHT / 2) as f32;
                self.paddle2_pos = (HEIGHT / 2) as f32;
                self.reset_ball_with_config(false, config); // Serve to player 1 (left)
            }
        }
    }

    /// Runs a full simulation episode between two individuals until a winner is decided.
    ///
    /// # Returns
    /// A nested tuple `((left_primary, left_secondary), (right_primary, right_secondary))`
    /// representing the two-component fitness score for each individual.
    ///
    /// # Algorithm
    /// 1.  Reset all game state variables.
    /// 2.  Loop until a max score is reached or a timeout occurs.
    /// 3.  Call `self.tick()` to advance the simulation by one step.
    /// 4.  Calculate and return the final fitness scores based on the configured fitness function.
    pub fn simulate<I: Individual>(
        &mut self,
        left: &I,
        right: &I,
        config: &Config,
    ) -> ((u32, u32), (u32, u32)) {
        self.reset();

        // # Teaching Note
        // The simulation runs for a maximum number of ticks. In the C++ version, this was
        // calculated based on game parameters to be a high number, effectively a timeout.
        // We'll use a fixed large number of ticks to serve the same purpose.
        let timelimit = 2 * 16 * MAX_SCORE as u32 * WIDTH as u32; // Generous timeout
        for _tick in 0..timelimit {
            if self.scores.0 >= MAX_SCORE || self.scores.1 >= MAX_SCORE {
                break;
            }
            self.tick(left, right, config);
        }

        // Calculate fitness based on the selected function
        match config.fitness_func {
            // C++ equivalent fitness: (primary, secondary) where primary is returns + shots
            // and secondary is wins. This requires a multi-objective sort in the population.
            FitnessFunc::CppEquivalent => {
                let left_wins = if self.scores.0 > self.scores.1 { 1 } else { 0 };
                let right_wins = if self.scores.1 > self.scores.0 { 1 } else { 0 };
                let left_primary = self.returns.0 + self.shots.0;
                let right_primary = self.returns.1 + self.shots.1;
                ((left_primary, left_wins), (right_primary, right_wins))
            }

            // The other fitness functions are kept for experimentation but adapted to the new
            // tuple return type. We'll use 0 for the secondary objective.
            FitnessFunc::Balanced => {
                let mut left_score = self.returns.0;
                let mut right_score = self.returns.1;
                if self.scores.0 > self.scores.1 {
                    left_score += 5;
                } else if self.scores.1 > self.scores.0 {
                    right_score += 5;
                }
                ((left_score, 0), (right_score, 0))
            }
            FitnessFunc::Performance => {
                let mut left_score = self.returns.0;
                let mut right_score = self.returns.1;
                if self.scores.0 >= MAX_SCORE {
                    left_score += 10; // Bonus for a decisive win
                }
                if self.scores.1 >= MAX_SCORE {
                    right_score += 10;
                }
                ((left_score, 0), (right_score, 0))
            }
        }
    }
}
