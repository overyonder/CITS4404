//! Keyboard controller for human players.
//!
//! This module provides a controller that allows human players to control paddles
//! using keyboard input (up/down arrow keys) at maximum velocity.

use crate::{config::Activation, engines::constants::*};
use crate::traits::Individual;

/// A controller that responds to keyboard input for human players.
///
/// # Teaching Note: Human-AI Interface Design
/// This controller enables human players to compete against AI agents, providing:
/// - **Direct control**: Immediate response to player input
/// - **Fair competition**: Uses same max velocity constraints as AI
/// - **Simple interface**: Up/down keys for intuitive control
/// - **Research value**: Allows comparison of human vs AI strategies
#[derive(Clone)]
pub struct KeyboardController {
    /// Current desired movement: -1.0 (down), 0.0 (none), 1.0 (up)
    pub movement: f32,
    /// Dummy weights to satisfy the Individual trait interface
    weights: [f32; TOTAL_WEIGHTS],
}

impl KeyboardController {
    /// Creates a new keyboard controller.
    pub fn new() -> Self {
        Self {
            movement: 0.0,
            weights: [0.0; TOTAL_WEIGHTS], // Dummy weights - not used for keyboard control
        }
    }
}

impl Individual for KeyboardController {
    /// Returns the movement command based on current keyboard state.
    /// Ignores the input state since this is direct human control.
    fn forward_propagate(&self, _input: &[f32; INPUT_SIZE], _activation: Activation) -> [f32; OUTPUT_SIZE] {
        [self.movement]
    }

    /// Crossover is not applicable for human controllers.
    fn crossover<R: rand::Rng>(&self, _other: &Self, _rng: &mut R) -> Self {
        Self::new()
    }

    /// Mutation is not applicable for human controllers.
    fn mutate<R: rand::Rng>(&mut self, _rng: &mut R, _config: &crate::Config) {
        // Human controllers don't mutate
    }

    /// Returns the dummy weights (keyboard controllers don't use weights).
    fn weights_as_slice(&self) -> &[f32] {
        &self.weights
    }

    /// Returns mutable access to dummy weights.
    fn weights_as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

impl Default for KeyboardController {
    fn default() -> Self {
        Self::new()
    }
} 