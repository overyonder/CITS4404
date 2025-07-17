use macroquad::input::{KeyCode, is_key_down};

pub const HEIGHT: f32 = 2.0;
pub const WIDTH: f32 = 1.0;
pub const PADDLE_LENGTH: f32 = HEIGHT / 8.0;
pub const PADDLE_WIDTH: f32 = PADDLE_LENGTH / 6.0;
pub const FRAME_RATE: f32 = 120.0;
pub const MAX_VEL: f32 = 1.0 / FRAME_RATE;

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
    pub left_player: Player,
    pub right_player: Player,
}

impl Game {
    pub fn new(left_player: Player, right_player: Player) -> Self {
        Self {
            state: [0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.5, 0.25],
            left_player,
            right_player,
        }
    }

    pub fn tick(&mut self) {
        self.state[1] = self.left_player.pass(&self) * MAX_VEL;
        self.state[3] = self.right_player.pass(&self) * MAX_VEL;
        self.state[0] += self.state[1];
        self.state[2] += self.state[3];
        self.state[4] += self.state[5];
        self.state[6] += self.state[7];
    }
}

/// A controller is a function that takes a gamestate and returns a decision.
trait Controller {
    fn pass(&self, state: &Game) -> f32;
}

/// A player is a controller that uses keyboard input to move the paddle.
pub struct Player {
    pub up_key: KeyCode,
    pub down_key: KeyCode,
}

impl Controller for Player {
    fn pass(&self, _: &Game) -> f32 {
        if is_key_down(self.up_key) {
            -1.0
        } else if is_key_down(self.down_key) {
            1.0
        } else {
            0.0
        }
    }
}
