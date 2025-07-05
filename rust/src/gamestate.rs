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
use rand::Rng;

/// Represents the full state of a Pong game simulation at a single point in time.
///
/// # Memory Layout
/// All fields are simple, stack-allocated `Copy` types. This makes the entire
/// `GameState` struct `Copy`-able, allowing for extremely cheap cloning. This is
/// beneficial in genetic algorithms where many simulations might be run in parallel
/// from a common starting state.
///
/// # Neural Network Input
/// The game state is converted into an 8-element array for the neural network:
/// 1.  Ball X position (normalized)
/// 2.  Ball Y position (normalized)
/// 3.  Ball X velocity (normalized)
/// 4.  Ball Y velocity (normalized)
/// 5.  Own paddle Y position (normalized)
/// 6.  Own paddle Y velocity (normalized)
/// 7.  Opponent paddle Y position (normalized)
/// 8.  Opponent paddle Y velocity (normalized)
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
}

impl GameState {
    /// Creates a new `GameState` with paddles and ball centered and scores zeroed.
    pub fn new() -> Self {
        let mut new_state = Self {
            paddle1_pos: (HEIGHT / 2) as f32,
            paddle1_vel: 0.0,
            paddle2_pos: (HEIGHT / 2) as f32,
            paddle2_vel: 0.0,
            ball_pos: (0.0, 0.0), // Position is set by reset_ball
            ball_vel: (0.0, 0.0), // Velocity is set by reset_ball
            scores: (0, 0),
            returns: (0, 0),
        };
        new_state.reset_ball(rand::thread_rng().gen()); // Set initial ball state
        new_state
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
            self.ball_pos.0 / WIDTH as f32,    // Ball X
            self.ball_pos.1 / HEIGHT as f32,   // Ball Y
            self.ball_vel.0 / BALL_MAX_SPEED,  // Ball Vel X
            self.ball_vel.1 / BALL_MAX_SPEED,  // Ball Vel Y
            self.paddle1_pos / HEIGHT as f32,  // Own Paddle Y
            self.paddle1_vel / PADDLE_MAX_VEL, // Own Paddle Vel Y
            self.paddle2_pos / HEIGHT as f32,  // Opponent Paddle Y
            self.paddle2_vel / PADDLE_MAX_VEL, // Opponent Paddle Vel Y
        ]
    }

    /// Prepares the 8-element input array for the **right paddle's** neural network.
    ///
    /// # Teaching Note
    /// To allow a single neural network to control both paddles, we can reuse the same
    /// network by simply flipping its perspective of the world. The ball's X position and
    /// velocity are inverted. This is a common and powerful technique in symmetric games.
    pub fn get_inputs_for_player2(&self) -> [f32; 8] {
        [
            (WIDTH as f32 - self.ball_pos.0) / WIDTH as f32, // Flipped Ball X
            self.ball_pos.1 / HEIGHT as f32,                 // Ball Y
            -self.ball_vel.0 / BALL_MAX_SPEED,               // Flipped Ball Vel X
            self.ball_vel.1 / BALL_MAX_SPEED,                // Ball Vel Y
            self.paddle2_pos / HEIGHT as f32,                // Own Paddle Y (is paddle2)
            self.paddle2_vel / PADDLE_MAX_VEL,               // Own Paddle Vel Y
            self.paddle1_pos / HEIGHT as f32,                // Opponent Paddle Y (is paddle1)
            self.paddle1_vel / PADDLE_MAX_VEL,               // Opponent Paddle Vel Y
        ]
    }

    /// Resets the ball to the center of the screen with a new random velocity.
    pub fn reset_ball(&mut self, serve_to_left_player: bool) {
        self.ball_pos = ((WIDTH / 2) as f32, (HEIGHT / 2) as f32);

        let mut rng = rand::thread_rng();
        // Aim the ball within a 45-degree cone towards the opponent.
        let angle = rng.gen_range(-std::f32::consts::FRAC_PI_4..=std::f32::consts::FRAC_PI_4);

        self.ball_vel.0 = BALL_INITIAL_SPEED * angle.cos();
        self.ball_vel.1 = BALL_INITIAL_SPEED * angle.sin();

        // If serving to the right player, the ball moves left (negative x velocity).
        if !serve_to_left_player {
            self.ball_vel.0 *= -1.0;
        }
    }

    /// Advances the game by one full tick during a training simulation.
    ///
    /// # Algorithm
    /// 1.  `update_paddles`: Get decisions from the neural networks and set paddle velocities.
    /// 2.  `advance_frame`: Update all positions and handle physics/collisions for the tick.
    pub fn tick<I: Individual>(&mut self, left: &I, right: &I, config: &Config) {
        self.update_paddles(left, right, config);
        self.advance_frame();
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
        self.paddle1_vel = (left_output[0] * PADDLE_MAX_VEL).clamp(-PADDLE_MAX_VEL, PADDLE_MAX_VEL);
        self.paddle2_vel =
            (right_output[0] * PADDLE_MAX_VEL).clamp(-PADDLE_MAX_VEL, PADDLE_MAX_VEL);
    }

    /// Advances the game state by one frame, applying physics and handling collisions.
    ///
    /// This function is used by both the training simulation (`tick`) and the interactive TUI.
    /// It assumes paddle velocities have already been set for the current frame.
    ///
    /// # Algorithm
    /// 1.  Update paddle positions based on their current velocities, clamping to screen bounds.
    /// 2.  Call `update_ball` to handle all ball-related physics for the frame.
    pub fn advance_frame(&mut self) {
        self.paddle1_pos =
            (self.paddle1_pos + self.paddle1_vel).clamp(MIN_PADDLE_POS, MAX_PADDLE_POS);
        self.paddle2_pos =
            (self.paddle2_pos + self.paddle2_vel).clamp(MIN_PADDLE_POS, MAX_PADDLE_POS);
        self.update_ball();
    }

    /// Updates the ball's position and handles all collisions.
    ///
    /// # Teaching Note
    /// This function contains the core physics logic. The order of operations is important:
    /// move, then check for collisions. Collision checks use the ball's radius for accuracy.
    /// The paddle collision logic includes imparting some of the paddle's velocity to the ball,
    /// creating more dynamic and interesting rallies.
    pub fn update_ball(&mut self) {
        self.ball_pos.0 += self.ball_vel.0;
        self.ball_pos.1 += self.ball_vel.1;

        // Top/Bottom wall collision
        if (self.ball_pos.1 - BALL_RADIUS <= 0.0 && self.ball_vel.1 < 0.0)
            || (self.ball_pos.1 + BALL_RADIUS >= HEIGHT as f32 && self.ball_vel.1 > 0.0)
        {
            self.ball_vel.1 *= -1.0;
        }

        // Paddle collision detection
        let paddle1_box = (PADDLE_WIDTH as f32, self.paddle1_pos);
        let paddle2_box = (WIDTH as f32 - PADDLE_WIDTH as f32, self.paddle2_pos);

        // Left paddle collision
        if self.ball_vel.0 < 0.0 && self.ball_pos.0 - BALL_RADIUS <= paddle1_box.0 {
            let paddle_top = paddle1_box.1 - PADDLE_HEIGHT as f32 / 2.0;
            let paddle_bottom = paddle1_box.1 + PADDLE_HEIGHT as f32 / 2.0;
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_vel.0 = self.ball_vel.0.abs(); // Reflect ball horizontally
                self.ball_vel.1 += self.paddle1_vel * 0.4; // Impart some paddle velocity
                self.returns.0 += 1;
            }
        // Right paddle collision
        } else if self.ball_vel.0 > 0.0 && self.ball_pos.0 + BALL_RADIUS >= paddle2_box.0 {
            let paddle_top = paddle2_box.1 - PADDLE_HEIGHT as f32 / 2.0;
            let paddle_bottom = paddle2_box.1 + PADDLE_HEIGHT as f32 / 2.0;
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_vel.0 = -self.ball_vel.0.abs(); // Reflect ball horizontally
                self.ball_vel.1 += self.paddle2_vel * 0.4; // Impart some paddle velocity
                self.returns.1 += 1;
            }
        }

        // Clamp ball speed to prevent it from getting too fast
        let speed = (self.ball_vel.0.powi(2) + self.ball_vel.1.powi(2)).sqrt();
        if speed > BALL_MAX_SPEED {
            self.ball_vel.0 = (self.ball_vel.0 / speed) * BALL_MAX_SPEED;
            self.ball_vel.1 = (self.ball_vel.1 / speed) * BALL_MAX_SPEED;
        }

        // Score detection (when ball goes past a paddle)
        if self.ball_pos.0 + BALL_RADIUS < 0.0 {
            self.scores.1 += 1;
            self.reset_ball(false); // Serve to player 1 (left)
        } else if self.ball_pos.0 - BALL_RADIUS > WIDTH as f32 {
            self.scores.0 += 1;
            self.reset_ball(true); // Serve to player 2 (right)
        }
    }

    /// Runs a full simulation episode between two individuals until a winner is decided.
    ///
    /// # Returns
    /// A tuple `(left_returns, right_returns)` representing the fitness score for each individual.
    ///
    /// # Algorithm
    /// 1.  Reset scores, returns, and the ball's position.
    /// 2.  Loop for a maximum number of ticks (e.g., 30 seconds worth).
    /// 3.  In each tick, check for a game-over condition (max score reached).
    /// 4.  Call `self.tick()` to advance the simulation by one step.
    /// 5.  Return the total number of successful returns for each player.
    pub fn simulate<I: Individual>(&mut self, left: &I, right: &I, config: &Config) -> (u32, u32) {
        self.scores = (0, 0);
        self.returns = (0, 0);
        self.reset_ball(rand::thread_rng().gen());

        // Run for a maximum of 30 seconds worth of ticks.
        for _tick in 0..(TICK_RATE as u32 * 30) {
            if self.scores.0 >= MAX_SCORE || self.scores.1 >= MAX_SCORE {
                break;
            }
            self.tick(left, right, config);
        }

        // Calculate fitness based on the selected function
        match config.fitness_func {
            // The original C++ fitness function was simply the number of returns.
            FitnessFunc::CppEquivalent => self.returns,

            // A balanced approach that rewards returns, but also winning.
            // A win is worth 5 returns.
            FitnessFunc::Balanced => {
                let mut left_score = self.returns.0;
                let mut right_score = self.returns.1;
                if self.scores.0 > self.scores.1 {
                    left_score += 5;
                } else if self.scores.1 > self.scores.0 {
                    right_score += 5;
                }
                (left_score, right_score)
            }

            // A performance-focused function that heavily rewards winning.
            FitnessFunc::Performance => {
                let mut left_score = self.returns.0;
                let mut right_score = self.returns.1;
                if self.scores.0 >= MAX_SCORE {
                    left_score += 10;
                }
                if self.scores.1 >= MAX_SCORE {
                    right_score += 10;
                }
                (left_score, right_score)
            }
        }
    }
}
