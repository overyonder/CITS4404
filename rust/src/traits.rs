use crate::config::EvolutionConfig;
use crate::constants::{INPUT_SIZE, OUTPUT_SIZE, TOTAL_WEIGHTS};
use rand::Rng;

use std::{
    fs,
    io::{self, Write},
};

/// A trait that defines the behavior of an individual in the population.
/// This allows for different implementations (e.g., stack-based, SIMD, GPU).
pub trait Individual: Default + Clone + Send + Sync {
    /// Provides the name of the individual type.
    ///
    /// # Returns
    /// A human-readable identifier for the representation.
    fn name() -> &'static str
    where
        Self: Sized;

    /// Computes the forward pass of the neural network.
    ///
    /// # Parameters
    /// - `input`: Array of floats representing the current game state.
    /// - `config`: Evolutionary parameters (mutation, crossover rate, etc.)
    /// # Returns
    /// Output value (e.g., paddle movement command).
    fn forward(&self, input: &[f32; INPUT_SIZE], config: &EvolutionConfig) -> [f32; OUTPUT_SIZE];

    /// Recombines this individual from two parents and applies mutation.
    ///
    /// # Parameters
    /// - `p1`, `p2`: Parent genomes.
    /// - `rng`: Random number generator.
    /// - `config`: Evolutionary parameters (mutation, crossover rate, etc.)
    fn recombine_from<R: Rng>(
        &mut self,
        p1: &Self,
        p2: &Self,
        rng: &mut R,
        config: &EvolutionConfig,
    );

    /// Provides a view of the weights as a flat slice for saving.
    ///
    /// # Returns
    /// A slice of floats representing the genome weights.
    fn weights_as_slice(&self) -> &[f32];

    /// Saves the individual's weights to a file.
    ///
    /// # Parameters
    /// - `path`: Path to save file.
    /// # Returns
    /// Result with I/O error if failed.
    fn save(&self, path: &str) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        let weights_slice = self.weights_as_slice();
        assert_eq!(weights_slice.len(), TOTAL_WEIGHTS);
        let weights_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights_slice.as_ptr() as *const u8,
                weights_slice.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(weights_bytes)
    }
}
