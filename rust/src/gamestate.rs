// IMPORTANT: Add `rand = "0.8"` to your [dependencies] in Cargo.toml
use crate::{constants::*, traits::Individual, config::EvolutionConfig};
use rand::Rng;

/// Represents the full state of a Pong game simulation.
///
/// # Fields
/// - `paddle1_pos`, `paddle2_pos`: Current positions of the paddles.
/// - `ball_pos`: Current position of the ball.
/// - `ball_vel`: Current velocity of the ball.
/// - `scores`: Current scores for each player.
/// - `returns`: Number of successful returns for each player.
///
/// # Memory
/// All fields are stack-allocated, small, and copyable for fast simulation.
///
/// # Neural Network Mapping
/// The neural net receives as input:
/// - Ball position (x, y)
/// - Ball velocity (x, y)
/// - Left paddle position, velocity
/// - Right paddle position, velocity
///
/// Output is a single value: move command for the paddle.
pub struct GameState {
    /// Current position of the left paddle.
    pub paddle1_pos: f32,
    /// Current position of the right paddle.
    pub paddle2_pos: f32,
    /// Current position of the ball.
    pub ball_pos: (f32, f32),
    /// Current velocity of the ball.
    pub ball_vel: (f32, f32),
    /// Current scores for each player.
    pub scores: (u8, u8),
    /// Number of successful returns for each player.
    pub returns: (u32, u32),
}

impl GameState {
    /// Creates a new game state with paddles and ball centered, zeroed scores.
    ///
    /// # Returns
    /// New `GameState` ready for simulation.
    pub fn new() -> Self {
        let mut new_state = Self {
            paddle1_pos: (LENGTH / 2) as f32,
            paddle2_pos: (LENGTH / 2) as f32,
            ball_pos: (0.0, 0.0), // Will be set by reset_ball
            ball_vel: (0.0, 0.0), // Will be set by reset_ball
            scores: (0, 0),
            returns: (0, 0),
        };
        new_state.reset_ball(false); // Initial ball state
        new_state
    }

    /// Resets the ball to the center with a random velocity.
    ///
    /// # Parameters
    /// - `right_serves`: Whether the right player serves the ball.
    fn reset_ball(&mut self, right_serves: bool) {
        self.ball_pos = ((WIDTH / 2) as f32, (LENGTH / 2) as f32);
        let mut rng = rand::thread_rng();
        // Give the ball a random angle, but not too vertical
        let angle = rng.gen_range(-std::f32::consts::FRAC_PI_4..=std::f32::consts::FRAC_PI_4);
        // Ball speed is relative to screen width and tick rate
        let speed = (WIDTH as f32) / (TICK_RATE as f32) * 0.5;
        self.ball_vel.0 = angle.cos() * speed;
        self.ball_vel.1 = angle.sin() * speed;
        if right_serves {
            self.ball_vel.0 *= -1.0;
        }
    }

    /// Advances the game state by one tick/frame.
    ///
    /// # Parameters
    /// - `left`, `right`: Individuals (neural nets) controlling each paddle.
    /// - `config`: Evolutionary/game parameters.
    pub fn tick<I: Individual>(&mut self, left: &I, right: &I, config: &EvolutionConfig) {
        self.update_paddles(left, right, config);
        self.update_ball();
    }

    /// Update the paddle positions based on neural net outputs.
    ///
    /// # Parameters
    /// - `left`, `right`: Individuals (neural nets) controlling each paddle.
    /// - `config`: Evolutionary/game parameters.
    fn update_paddles<I: Individual>(&mut self, left: &I, right: &I, config: &EvolutionConfig) {
        // --- Left Paddle ---
        let mut left_input = [0.0; INPUT_SIZE];
        left_input[0] = self.ball_pos.0 / WIDTH as f32; // Ball X
        left_input[1] = self.ball_pos.1 / LENGTH as f32; // Ball Y
        left_input[2] = self.ball_vel.0; // Ball Vel X
        left_input[3] = self.ball_vel.1; // Ball Vel Y
        left_input[4] = self.paddle1_pos / LENGTH as f32; // Own paddle Y
        left_input[5] = self.paddle2_pos / LENGTH as f32; // Opponent paddle Y
        let paddle_out = left.forward(&left_input, config);
        let target_vel = paddle_out[0] * PADDLE_MAX_VEL;
        self.paddle1_pos = (self.paddle1_pos + target_vel).clamp(MIN_PADDLE_POS, MAX_PADDLE_POS);

        // --- Right Paddle ---
        // Inputs are from the perspective of the right paddle
        let mut right_input = [0.0; INPUT_SIZE];
        right_input[0] = 1.0 - (self.ball_pos.0 / WIDTH as f32); // Invert X for opponent
        right_input[1] = self.ball_pos.1 / LENGTH as f32;
        right_input[2] = -self.ball_vel.0; // Invert vel X
        right_input[3] = self.ball_vel.1;
        right_input[4] = self.paddle2_pos / LENGTH as f32; // Own paddle Y
        right_input[5] = self.paddle1_pos / LENGTH as f32; // Opponent paddle Y
        let paddle_out = right.forward(&right_input, config);
        let target_vel = paddle_out[0] * PADDLE_MAX_VEL;
        self.paddle2_pos = (self.paddle2_pos + target_vel).clamp(MIN_PADDLE_POS, MAX_PADDLE_POS);
    }

    /// Update ball position and velocity, handling collisions.
    fn update_ball(&mut self) {
        self.ball_pos.0 += self.ball_vel.0;
        self.ball_pos.1 += self.ball_vel.1;

        // Wall collision (top/bottom)
        if self.ball_pos.1 <= 0.0 || self.ball_pos.1 >= LENGTH as f32 {
            self.ball_vel.1 *= -1.0;
        }

        // Paddle collision
        let ball_in_left_paddle_range = self.ball_pos.0 <= PADDLE_WIDTH as f32;
        let ball_in_right_paddle_range = self.ball_pos.0 >= WIDTH as f32 - PADDLE_WIDTH as f32;

        if ball_in_left_paddle_range {
            let paddle_top = self.paddle1_pos - PADDLE_HEIGHT as f32 / 2.0;
            let paddle_bottom = self.paddle1_pos + PADDLE_HEIGHT as f32 / 2.0;
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_vel.0 *= -1.0;
                self.returns.0 += 1;
            }
        } else if ball_in_right_paddle_range {
            let paddle_top = self.paddle2_pos - PADDLE_HEIGHT as f32 / 2.0;
            let paddle_bottom = self.paddle2_pos + PADDLE_HEIGHT as f32 / 2.0;
            if self.ball_pos.1 >= paddle_top && self.ball_pos.1 <= paddle_bottom {
                self.ball_vel.0 *= -1.0;
                self.returns.1 += 1;
            }
        }

        // Score condition
        if self.ball_pos.0 < 0.0 {
            self.scores.1 += 1;
            self.reset_ball(false); // Left player serves
        } else if self.ball_pos.0 > WIDTH as f32 {
            self.scores.0 += 1;
            self.reset_ball(true); // Right player serves
        }
    }

    /// Run a full simulation episode for two individuals.
    /// Returns the number of successful returns for (left, right).
    ///
    /// # Parameters
    /// - `left`, `right`: Individuals (neural nets) controlling each paddle.
    /// - `config`: Evolutionary/game parameters.
    ///
    /// # Returns
    /// Tuple: (returns_left, returns_right) — Number of successful returns for each player.
    ///
    /// # Algorithm
    /// - For each timestep:
    ///   - Feed current game state to each individual's neural net.
    ///   - Update paddle velocities/positions per output.
    ///   - Update ball position, handle collisions (walls, paddles).
    ///   - Increment scores if ball passes a paddle.
    ///   - End after max steps or score.
    pub fn simulate<I: Individual>(&mut self, left: &I, right: &I, config: &EvolutionConfig) -> (u32, u32) {
        self.scores = (0, 0);
        self.returns = (0, 0);
        self.reset_ball(rand::thread_rng().gen());

        // Run for a max of 30 seconds
        for _tick in 0..(TICK_RATE as u32 * 30) {
            if self.scores.0 >= MAX_SCORE || self.scores.1 >= MAX_SCORE {
                break;
            }
            self.tick(left, right, config);
        }
        self.returns
    }
}
