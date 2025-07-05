//! Simulation state and logic for Pong visualization in the TUI.
use crate::{
    config::Config,
    constants::PADDLE_MAX_VEL,
    engines::HeapIndividual, // Using HeapIndividual as a concrete type for the simulation brain
    gamestate::GameState,
    traits::Individual,
};

pub struct SimulationState {
    pub game: GameState,
    p1_brain: HeapIndividual,
    p2_brain: HeapIndividual, // The AI will play against itself
}

impl SimulationState {
    /// Creates a new simulation state from a trained genome.
    pub fn new(best_genome: Vec<f32>) -> Self {
        // The `weights` field of `HeapIndividual` is public, so we can construct it directly.
        let p1_brain = HeapIndividual {
            weights: best_genome.clone(),
        };
        let p2_brain = HeapIndividual {
            weights: best_genome,
        };

        Self {
            game: GameState::new(),
            p1_brain,
            p2_brain,
        }
    }

    /// Advance the simulation by one tick/frame, using the neural network to control the paddles.
    pub fn step(&mut self, config: &Config) {
        // Get inputs for both players from the current game state.
        let p1_inputs = self.game.get_inputs_for_player1();
        let p2_inputs = self.game.get_inputs_for_player2();

        // Get actions from the neural networks.
        let p1_outputs = self
            .p1_brain
            .forward_propagate(&p1_inputs, config.activation);
        let p2_outputs = self
            .p2_brain
            .forward_propagate(&p2_inputs, config.activation);

        // Update paddle velocities based on network outputs.
        // The output is in the range [-1, 1], which we scale by the max paddle velocity constant.
        self.game.paddle1_vel = p1_outputs[0] * PADDLE_MAX_VEL;
        self.game.paddle2_vel = p2_outputs[0] * PADDLE_MAX_VEL;

        // Advance the game state by one frame.
        self.game.advance_frame();
    }
}
