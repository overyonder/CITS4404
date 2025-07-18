use rand_distr::Distribution;

use crate::nn::Individual;

use super::controllers::Controller;

pub const DEF_MEAN: f32 = 0.;
pub const BALL_START_VEL: f32 = 0.01;
pub const DEF_STD_DEV: f32 = 0.05;

pub const HEIGHT: f32 = 1.;
pub const WIDTH: f32 = 0.66;

pub const PADDLE_LENGTH: f32 = HEIGHT / 8.;
pub const PADDLE_WIDTH: f32 = PADDLE_LENGTH / 10.;

pub const FRAME_RATE: f32 = 60.;
pub const MAX_PADDLE_VEL: f32 = 1. / FRAME_RATE;
pub const MAX_BALL_Y_VEL: f32 = MAX_PADDLE_VEL * 1.25 * FRAME_RATE;

pub const MIN_Y: f32 = PADDLE_LENGTH / 2.;
pub const MAX_Y: f32 = 1. - PADDLE_LENGTH / 2.;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Gamestate. Contains a complete representation of a single match point.
pub struct Game {
    /// - 0 = left_pos,      // [-0.5, 0.5]
    /// - 1 = left_vel,      // [-1, 1]
    /// - 2 = right_pos,     // [-0.5, 0.5]
    /// - 3 = right_vel,     // [-1, 1]
    /// - 4 = ball_pos_x,    // [-0.5, 0.5]
    /// - 5 = ball_vel_x,    // [-1, 1]
    /// - 6 = ball_pos_y,    // [-0.5, 0.5]
    /// - 7 = ball_vel_y,    // [-1, 1]
    /// - 8 = bias           // 1.0
    pub state: [f32; 9],
}

impl Default for Game {
    fn default() -> Self {
        Self {
            state: [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                BALL_START_VEL / MAX_PADDLE_VEL,
                0.0,
                BALL_START_VEL / MAX_PADDLE_VEL,
                1.0,
            ],
        }
    }
}

impl Game {
    pub fn reset(&mut self, serve: Side) {
        self.state = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            BALL_START_VEL / MAX_PADDLE_VEL,
            0.0,
            BALL_START_VEL / MAX_PADDLE_VEL,
            1.0,
        ];
        if serve == Side::Left {
            self.state[5] = BALL_START_VEL / MAX_PADDLE_VEL;
        } else {
            self.state[5] = -BALL_START_VEL / MAX_PADDLE_VEL;
        }
    }

    pub fn run_until(
        &mut self,
        left_controller: &Individual,
        right_controller: &Individual,
        serve: Side,
    ) -> (Side, usize) {
        let max_ticks = (FRAME_RATE * 120.) as usize;
        let mut ticks = 0;
        self.reset(serve);
        loop {
            if ticks >= max_ticks {
                return (Side::Left, ticks);
            }
            let result = self.tick(left_controller, right_controller);
            if let Some(winner) = result {
                return (winner, ticks);
            }
            ticks += 1;
        }
    }

    pub fn tick<T: Controller, U: Controller>(
        &mut self,
        left_controller: &T,
        right_controller: &U,
    ) -> Option<Side> {
        // Left update
        *self.left_vel_mut() = left_controller.pass(&self.state);
        *self.left_pos_mut() += self.left_vel() * MAX_PADDLE_VEL / HEIGHT;
        if self.left_pos() <= -0.5 + PADDLE_LENGTH / 2. {
            *self.left_pos_mut() = -0.5 + PADDLE_LENGTH / 2.;
            *self.left_vel_mut() = self.left_vel_mut().clamp(0., 1.);
        } else if self.left_pos() >= 0.5 - PADDLE_LENGTH / 2. {
            *self.left_pos_mut() = 0.5 - PADDLE_LENGTH / 2.;
            *self.left_vel_mut() = self.left_vel_mut().clamp(-1., 0.);
        }

        // Right update
        // Flip state for right paddle: x positions and velocities are scaled by WIDTH
        let flipped_state = [
            self.state[2],  // right_pos
            self.state[3],  // right_vel
            self.state[0],  // left_pos
            self.state[1],  // left_vel
            -self.state[4], // ball_pos_x (negate, but still in [-0.5, 0.5])
            -self.state[5], // ball_vel_x
            self.state[6],  // ball_pos_y
            self.state[7],  // ball_vel_y
            self.state[8],  // bias
        ];
        *self.right_vel_mut() = right_controller.pass(&flipped_state);
        *self.right_pos_mut() += self.right_vel() * MAX_PADDLE_VEL / HEIGHT;
        if self.right_pos() <= -0.5 + PADDLE_LENGTH / 2. {
            *self.right_pos_mut() = -0.5 + PADDLE_LENGTH / 2.;
            *self.right_vel_mut() = self.right_vel_mut().clamp(0., 1.);
        } else if self.right_pos() >= 0.5 - PADDLE_LENGTH / 2. {
            *self.right_pos_mut() = 0.5 - PADDLE_LENGTH / 2.;
            *self.right_vel_mut() = self.right_vel_mut().clamp(-1., 0.);
        }

        // Ball update: scale velocity by WIDTH for x
        *self.ball_pos_x_mut() += self.ball_vel_x() * MAX_PADDLE_VEL / WIDTH;
        *self.ball_pos_y_mut() += self.ball_vel_y() * MAX_PADDLE_VEL / HEIGHT;

        // Ball to score zone (normalized boundaries)
        let left_x_bound = -0.5 + PADDLE_WIDTH / WIDTH;
        let right_x_bound = 0.5 - PADDLE_WIDTH / WIDTH;
        if self.ball_pos_x() <= left_x_bound {
            // Check for rebound
            if self.left_pos() - PADDLE_LENGTH / 2. <= self.ball_pos_y()
                && self.ball_pos_y() <= self.left_pos() + PADDLE_LENGTH / 2.
            {
                *self.ball_pos_x_mut() = left_x_bound + f32::EPSILON;
                *self.ball_vel_x_mut() = -self.ball_vel_x();
                *self.ball_vel_y_mut() += self.left_vel() * 0.25
                    + rand_distr::Normal::new(DEF_MEAN, DEF_STD_DEV)
                        .unwrap()
                        .sample(&mut rand::rng())
                        * BALL_START_VEL
                        / MAX_PADDLE_VEL;
                *self.ball_vel_y_mut() =
                    self.ball_vel_y().clamp(-MAX_BALL_Y_VEL, MAX_BALL_Y_VEL);
            } else {
                self.reset(Side::Left);
                return Some(Side::Right);
            }
        } else if self.ball_pos_x() >= right_x_bound {
            // Check for rebound
            if self.right_pos() - PADDLE_LENGTH / 2. <= self.ball_pos_y()
                && self.ball_pos_y() <= self.right_pos() + PADDLE_LENGTH / 2.
            {
                *self.ball_pos_x_mut() = right_x_bound - f32::EPSILON;
                *self.ball_vel_x_mut() = -self.ball_vel_x();
                *self.ball_vel_y_mut() += self.right_vel() * 0.25
                    + rand_distr::Normal::new(DEF_MEAN, DEF_STD_DEV)
                        .unwrap()
                        .sample(&mut rand::rng())
                        * BALL_START_VEL
                        / MAX_PADDLE_VEL;
                *self.ball_vel_y_mut() =
                    self.ball_vel_y().clamp(-MAX_BALL_Y_VEL, MAX_BALL_Y_VEL);
            } else {
                self.reset(Side::Right);
                return Some(Side::Left);
            }
        }

        // Ball to ceil/floor
        if self.ball_pos_y() < -0.5 {
            *self.ball_pos_y_mut() = -0.5;
            *self.ball_vel_y_mut() = -self.ball_vel_y();
        } else if self.ball_pos_y() > 0.5 {
            *self.ball_pos_y_mut() = 0.5;
            *self.ball_vel_y_mut() = -self.ball_vel_y();
        }

        None
    }
}

// Separate impl block for all inline getters
impl Game {
    #[inline(always)]
    pub fn left_pos(&self) -> f32 {
        self.state[0]
    }
    #[inline(always)]
    pub fn left_pos_mut(&mut self) -> &mut f32 {
        &mut self.state[0]
    }
    #[inline(always)]
    pub fn left_vel(&self) -> f32 {
        self.state[1]
    }
    #[inline(always)]
    pub fn left_vel_mut(&mut self) -> &mut f32 {
        &mut self.state[1]
    }

    #[inline(always)]
    pub fn right_pos(&self) -> f32 {
        self.state[2]
    }
    #[inline(always)]
    pub fn right_pos_mut(&mut self) -> &mut f32 {
        &mut self.state[2]
    }
    #[inline(always)]
    pub fn right_vel(&self) -> f32 {
        self.state[3]
    }
    #[inline(always)]
    pub fn right_vel_mut(&mut self) -> &mut f32 {
        &mut self.state[3]
    }

    #[inline(always)]
    pub fn ball_pos_x(&self) -> f32 {
        self.state[4]
    }
    #[inline(always)]
    pub fn ball_pos_x_mut(&mut self) -> &mut f32 {
        &mut self.state[4]
    }
    #[inline(always)]
    pub fn ball_vel_x(&self) -> f32 {
        self.state[5]
    }
    #[inline(always)]
    pub fn ball_vel_x_mut(&mut self) -> &mut f32 {
        &mut self.state[5]
    }

    #[inline(always)]
    pub fn ball_pos_y(&self) -> f32 {
        self.state[6]
    }
    #[inline(always)]
    pub fn ball_pos_y_mut(&mut self) -> &mut f32 {
        &mut self.state[6]
    }
    #[inline(always)]
    pub fn ball_vel_y(&self) -> f32 {
        self.state[7]
    }
    #[inline(always)]
    pub fn ball_vel_y_mut(&mut self) -> &mut f32 {
        &mut self.state[7]
    }
}
