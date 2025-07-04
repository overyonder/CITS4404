use crate::constants::{
    BALL_START_VEL, LENGTH, MAX_POSITION, PADDLE_HEIGHT, PADDLE_MAX_VEL, PADDLE_WIDTH, WIDTH,
};
use crate::game::Ball;
use crate::net::Net;
use serde_json;
use std::fs;

#[derive(Debug, Clone)]
pub struct Paddle {
    pub position: u16,
    pub velocity: i16,
    pub net: Net,
}

impl Paddle {
    pub fn new_with_net(net: Net) -> Self {
        Self {
            position: (LENGTH - PADDLE_HEIGHT) / 2,
            velocity: 0,
            net,
        }
    }

    pub fn from_best_net(path: &str) -> Self {
        match fs::read_to_string(path) {
            Ok(data) => {
                println!("Loading best net from '{}'...", path);
                let net: Net = serde_json::from_str(&data).expect("Failed to deserialize net");
                Self::new_with_net(net)
            }
            Err(_) => {
                println!("Could not find best net file at '{}', creating a new default paddle.", path);
                Self::new()
            }
        }
    }

    pub fn new() -> Self {
        Self {
            position: (LENGTH - PADDLE_HEIGHT) / 2,
            velocity: 0,
            net: Net::new(),
        }
    }

    pub fn update_velocity(&mut self, opponent_y: f64, ball: &Ball, player_index: usize) {
        // 1. Construct the 8-element input array for the neural network.
        let paddle_x = if player_index == 0 {
            PADDLE_WIDTH as f64
        } else {
            (WIDTH - PADDLE_WIDTH) as f64
        };

        let inputs = [
            self.position as f64 / MAX_POSITION as f64,
            opponent_y / MAX_POSITION as f64,
            ball.position[0] / WIDTH as f64,
            ball.position[1] / LENGTH as f64,
            ball.velocity[0] / BALL_START_VEL[0],
            ball.velocity[1] / BALL_START_VEL[1],
            (ball.position[0] - paddle_x) / WIDTH as f64,
            (ball.position[1] - self.position as f64) / LENGTH as f64,
        ];

        // 2. Call the forward_propagate method of the network.
        let output = self.net.forward_propagate(&inputs);

        // 3. Interpret the output to set the paddle's velocity.
        self.velocity = (output * PADDLE_MAX_VEL as f64) as i16;
        //    The network output is typically in the range -1.0 to 1.0. Scale it by `PADDLE_MAX_VEL`.
        //    `self.velocity = (output * PADDLE_MAX_VEL as f64) as i16;`
    }

    pub fn update_position(&mut self) {
        // Update paddle position using its velocity, preventing overflow.
        self.position = self.position.saturating_add_signed(self.velocity);

        // Clamp the position to ensure it stays within the game boundaries.
        self.position = self.position.clamp(0, MAX_POSITION);
    }
}
