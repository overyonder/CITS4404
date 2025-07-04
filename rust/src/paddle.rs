use crate::constants::{LENGTH, MAX_POSITION, PADDLE_HEIGHT, PADDLE_MAX_VEL, WIDTH};
use crate::game::Ball;
use crate::net::Net;

#[derive(Debug, Clone)]
pub struct Paddle {
    pub position: u16,
    pub velocity: i16,
    pub net: Net,
}

impl Paddle {
    pub fn new() -> Self {
        Self {
            position: (LENGTH - PADDLE_HEIGHT) / 2,
            velocity: 0,
            net: Net::new(),
        }
    }

    pub fn update_velocity(&mut self, opponent_y: f64, ball: &Ball, player_index: usize) {
        // Create the 8-element input vector for the neural network by normalizing game state.
        let inputs = {
            let own_y_norm = self.position as f64 / LENGTH as f64;
            let opponent_y_norm = opponent_y / LENGTH as f64;

            // The perspective of the ball's X position and velocity is flipped for the right paddle.
            let (ball_x_norm, ball_vx_norm) = if player_index == 0 {
                // Left paddle's perspective
                (
                    ball.position[0] / WIDTH as f64,
                    ball.velocity[0] / WIDTH as f64,
                )
            } else {
                // Right paddle's perspective (flipped)
                (
                    (WIDTH as f64 - ball.position[0]) / WIDTH as f64,
                    -ball.velocity[0] / WIDTH as f64,
                )
            };

            let ball_y_norm = ball.position[1] / LENGTH as f64;
            let ball_vy_norm = ball.velocity[1] / LENGTH as f64;

            // Two extra inputs to give the network more context.
            let own_y_from_center =
                (self.position as f64 - LENGTH as f64 / 2.0) / (LENGTH as f64 / 2.0);
            let ball_dist_y = (self.position as f64 - ball.position[1]) / LENGTH as f64;

            [
                own_y_norm,
                opponent_y_norm,
                ball_x_norm,
                ball_y_norm,
                ball_vx_norm,
                ball_vy_norm,
                own_y_from_center,
                ball_dist_y,
            ]
        };

        // Get the network's output (-1.0 to 1.0).
        let output = self.net.forward_propagate(&inputs);

        // Map the output to the paddle's velocity and cast to i16.
        self.velocity = (output * PADDLE_MAX_VEL as f64) as i16;
    }

    pub fn update_position(&mut self) {
        // Update paddle position using its velocity, preventing overflow.
        self.position = self.position.saturating_add_signed(self.velocity);

        // Clamp the position to ensure it stays within the game boundaries.
        self.position = self.position.clamp(0, MAX_POSITION);
    }

    pub fn mutate(&mut self) {
        self.net.mutate();
    }
}
