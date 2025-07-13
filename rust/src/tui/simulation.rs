//! Simulation state and logic for Pong visualization in the TUI.
use crate::{
    config::Config,
    engines::StackIndividual, // Using StackIndividual as a concrete type for the simulation brain
    gamestate::{constants::PADDLE_MAX_VEL, GameState},
    traits::Individual,
};

/// Holds the state for a single simulation match.
pub struct SimulationState {
    pub left_player: StackIndividual,
    pub right_player: StackIndividual,
    pub left_config: Config,
    pub right_config: Config,
    pub game_state: GameState,
}

impl SimulationState {
    /// Creates a new simulation with two specified individuals (by their weights).
    pub fn new(
        left_weights: Vec<f32>,
        right_weights: Vec<f32>,
        left_config: Config,
        right_config: Config,
    ) -> Self {
        Self {
            left_player: StackIndividual {
                weights: left_weights.try_into().expect("Invalid weights array size"),
            },
            right_player: StackIndividual {
                weights: right_weights.try_into().expect("Invalid weights array size"),
            },
            left_config,
            right_config,
            game_state: GameState::new(),
        }
    }

    /// Advances the simulation by one tick.
    pub fn step(&mut self, config: &Config) {
        // Get inputs for both players from the current game state.
        let p1_inputs = self.game_state.get_inputs_for_player1();
        let p2_inputs = self.game_state.get_inputs_for_player2();

        // Get actions from the neural networks.
        let p1_outputs = self
            .left_player
            .forward_propagate(&p1_inputs, config.activation);
        let p2_outputs = self
            .right_player
            .forward_propagate(&p2_inputs, config.activation);

        // Update paddle velocities based on network outputs.
        // The output is in the range [-1, 1], which we scale by the max paddle velocity constant.
        // Note: Right paddle output is negated to match C++ version behavior
        self.game_state.paddle1_vel = p1_outputs[0] * PADDLE_MAX_VEL;
        self.game_state.paddle2_vel = -p2_outputs[0] * PADDLE_MAX_VEL;

        // Advance the game state by one frame.
        self.game_state.advance_frame(config);
    }
}
