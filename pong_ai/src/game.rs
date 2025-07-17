const HEIGHT: f32 = 800.0;
const WIDTH: f32 = 400.0;
const PADDLE_LENGTH: f32 = HEIGHT/8.0;


/// Gamestate. Contains a complete representation of a single match point.
struct GameState {
    // left_pos, right_pos, ball_vel_y, ball_pos_x, ball_pos_y, ball_vel_x, left_vel, right_vel
    inputs: [f32;8]
}

/// A decision is a float from -1..=1 indicating the paddle moving down or up at it's max speed
struct Decision {
    vel: f32,
}

trait Controller {
    fn pass(state: &GameState) -> Decision;
}

