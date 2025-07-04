use crate::constants::*;
use crate::paddle::Paddle;

#[derive(Debug)]
pub struct Ball {
    // Using f64 for smooth, sub-pixel movement, matching the C++ version.
    pub position: [f64; 2],
    pub velocity: [f64; 2],
}

impl Ball {
    pub fn new() -> Self {
        Self {
            position: [WIDTH as f64 / 2.0, LENGTH as f64 / 2.0],
            velocity: BALL_START_VEL,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub paddles: [Paddle; 2],
    pub ball: Ball,
    pub score: [u8; 2],
    pub returns: [u16; 2],
    pub shots: [u16; 2],
    pub ticks: u64,
}

impl GameState {
    pub fn new() -> Self {
        Self {
            paddles: [Paddle::new(), Paddle::new()],
            ball: Ball::new(),
            score: [0, 0],
            returns: [0, 0],
            shots: [0, 0],
            ticks: 0,
        }
    }

    pub fn tick(&mut self) {
        // This function advances the game by one frame.
        self.update_paddles();
        self.update_ball();

        // Check if a point was scored.
        let mut scored = false;
        if self.ball.position[0] < 0.0 {
            self.score[1] += 1; // Right player scores
            scored = true;
        } else if self.ball.position[0] > WIDTH as f64 {
            self.score[0] += 1; // Left player scores
            scored = true;
        }

        if scored {
            // Reset ball and paddles to center positions.
            self.ball = Ball::new();
            self.paddles = [Paddle::new(), Paddle::new()];
        }

        self.ticks += 1;
    }

    fn update_paddles(&mut self) {
        // To satisfy the borrow checker, we get the positions before the mutable borrows.
        let p0_pos = self.paddles[0].position as f64;
        let p1_pos = self.paddles[1].position as f64;
        let ball = &self.ball;

        // Update velocities based on network output.
        self.paddles[0].update_velocity(p1_pos, ball, 0);
        self.paddles[1].update_velocity(p0_pos, ball, 1);

        // Apply the new velocities to the paddles' positions.
        self.paddles[0].update_position();
        self.paddles[1].update_position();
    }

    fn update_ball(&mut self) {
        // This function handles ball movement and collisions.

        // 1. Check for collision with top/bottom walls.
        let next_y = self.ball.position[1] + self.ball.velocity[1];
        if next_y < 0.0 || next_y > LENGTH as f64 {
            self.ball.velocity[1] *= -1.0;
        }

        // 2. Check for collision with left paddle.
        let next_x = self.ball.position[0] + self.ball.velocity[0];
        let paddle_top = self.paddles[0].position as f64 - PADDLE_HEIGHT as f64 / 2.0;
        let paddle_bottom = self.paddles[0].position as f64 + PADDLE_HEIGHT as f64 / 2.0;

        if next_x < PADDLE_WIDTH as f64 && self.ball.position[1] >= paddle_top && self.ball.position[1] <= paddle_bottom {
            self.ball.velocity[0] *= -1.0;
            // Optional: Add some of the paddle's velocity for spin.
            // self.ball.velocity[1] += self.left_paddle.velocity as f64 * 0.1;
            self.returns[0] += 1;
        }

        // 3. Check for collision with right paddle.
        let paddle_top = self.paddles[1].position as f64 - PADDLE_HEIGHT as f64 / 2.0;
        let paddle_bottom = self.paddles[1].position as f64 + PADDLE_HEIGHT as f64 / 2.0;

        if next_x > (WIDTH - PADDLE_WIDTH) as f64 && self.ball.position[1] >= paddle_top && self.ball.position[1] <= paddle_bottom {
            self.ball.velocity[0] *= -1.0;
            self.returns[1] += 1;
        }

        // 4. Update ball's position.
        self.ball.position[0] += self.ball.velocity[0];
        self.ball.position[1] += self.ball.velocity[1];
    }
}
