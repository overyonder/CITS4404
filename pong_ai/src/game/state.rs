use rand_distr::Distribution;

use super::controllers::Controller;

pub const DEFAULTS: [f32; 8] = [0.5, 0., 0.5, 0., 0.5, BALL_START_VEL, 0.5, BALL_START_VEL];
pub const DEF_MEAN: f32 = 0.;
pub const BALL_START_VEL: f32 = 0.01;
pub const DEF_STD_DEV: f32 = 0.05;

pub const HEIGHT: f32 = 1.;
pub const WIDTH: f32 = 0.66;

pub const PADDLE_LENGTH: f32 = HEIGHT / 8.;
pub const PADDLE_WIDTH: f32 = PADDLE_LENGTH / 10.;

pub const FRAME_RATE: f32 = 60.;
pub const MAX_VEL: f32 = 1. / FRAME_RATE;

pub const MIN_Y: f32 = PADDLE_LENGTH / 2.;
pub const MAX_Y: f32 = 1. - PADDLE_LENGTH / 2.;

#[derive(PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Gamestate. Contains a complete representation of a single match point.
pub struct Game {
    /// - 0 = left_pos,
    /// - 1 = left_vel,
    /// - 2 = right_pos,
    /// - 3 = right_vel,
    /// - 4 = ball_pos_x,
    /// - 5 = ball_vel_x,
    /// - 6 = ball_pos_y,
    /// - 7 = ball_vel_y
    pub state: [f32; 8],
}

impl Default for Game {
    fn default() -> Self {
        Self { state: DEFAULTS }
    }
}

impl Game {
    pub fn reset(&mut self, serve: Side) {
        self.state = DEFAULTS;
        if serve == Side::Left {
            self.state[5] = BALL_START_VEL;
        } else {
            self.state[5] = -BALL_START_VEL;
        }
    }

    pub fn run_until<T: Controller, U: Controller>(
        &mut self,
        left_controller: &T,
        right_controller: &U
    ) -> Side {
        let mut result = None;
        while result.is_none() {
            result = self.tick(left_controller, right_controller);
        }
        return result.unwrap();
    }

    pub fn tick<T: Controller, U: Controller>(
        &mut self,
        left_controller: &T,
        right_controller: &U
    ) -> Option<Side> {
        // Left update
        *self.left_vel_mut() = left_controller.pass(&self.state) * MAX_VEL;
        *self.left_pos_mut() += self.left_vel();
        if self.left_pos() <= MIN_Y - f32::EPSILON {
            *self.left_pos_mut() = MIN_Y;
            *self.left_vel_mut() = self.left_vel_mut().clamp(0., MAX_VEL);
        } else if self.left_pos() >= MAX_Y + f32::EPSILON {
            *self.left_pos_mut() = MAX_Y;
            *self.left_vel_mut() = self.left_vel_mut().clamp(-MAX_VEL, 0.);
        }

        // Right update
        *self.right_vel_mut() = right_controller.pass(&self.state) * MAX_VEL;
        *self.right_pos_mut() += self.right_vel();
        if self.right_pos() <= MIN_Y - f32::EPSILON {
            *self.right_pos_mut() = MIN_Y;
            *self.right_vel_mut() = self.right_vel_mut().clamp(0., MAX_VEL);
        } else if self.right_pos() >= MAX_Y + f32::EPSILON {
            *self.right_pos_mut() = MAX_Y;
            *self.right_vel_mut() = self.right_vel_mut().clamp(-MAX_VEL, 0.);
        }

        // Ball update
        *self.ball_pos_x_mut() += self.ball_vel_x();
        *self.ball_pos_y_mut() += self.ball_vel_y();

        // Ball to score zone
        if self.ball_pos_x() <= PADDLE_WIDTH - f32::EPSILON {
            // Check for rebound
            if self.left_pos() - PADDLE_LENGTH / 2. <= self.ball_pos_y()
                && self.ball_pos_y() <= self.left_pos() + PADDLE_LENGTH / 2.
            {
                *self.ball_pos_x_mut() = PADDLE_WIDTH;
                *self.ball_vel_x_mut() = -self.ball_vel_x();
                *self.ball_vel_y_mut() += self.left_vel() * 0.25
                    + rand_distr::Normal::new(DEF_MEAN, DEF_STD_DEV)
                        .unwrap()
                        .sample(&mut rand::rng())
                        * BALL_START_VEL;
            } else {
                // Otherwise score and reset
                self.reset(Side::Left);
                return Some(Side::Right);
            }
        } else if self.ball_pos_x() >= WIDTH - PADDLE_WIDTH + f32::EPSILON {
            // Check for rebound
            if self.right_pos() - PADDLE_LENGTH / 2. <= self.ball_pos_y()
                && self.ball_pos_y() <= self.right_pos() + PADDLE_LENGTH / 2.
            {
                *self.ball_pos_x_mut() = WIDTH - PADDLE_WIDTH;
                *self.ball_vel_x_mut() = -self.ball_vel_x();
                *self.ball_vel_y_mut() += self.right_vel() * 0.25
                    + rand_distr::Normal::new(DEF_MEAN, DEF_STD_DEV)
                        .unwrap()
                        .sample(&mut rand::rng())
                        * BALL_START_VEL;
            } else {
                // Otherwise score and reset
                self.reset(Side::Right);
                return Some(Side::Left);
            }
        }

        // Ball to ceil/floor
        if self.ball_pos_y() < 0. {
            *self.ball_pos_y_mut() = 0.;
            *self.ball_vel_y_mut() = -self.ball_vel_y();
        } else if self.ball_pos_y() >= 1. {
            *self.ball_pos_y_mut() = 1.;
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
