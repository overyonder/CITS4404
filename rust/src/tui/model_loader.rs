//! Handles the loading of trained models from files.
//!
//! # Teaching Note: Separation of Concerns
//! This module is responsible solely for model persistence, demonstrating the
//! **Single Responsibility Principle**. It's engine-agnostic, meaning the same
//! loading logic works regardless of whether the model will be used with CPU,
//! GPU, or other neural network backends.

use crate::config::Config;
use crate::engines::constants::TOTAL_WEIGHTS;
use crate::traits::SerializableIndividual;
use std::path::Path;

/// Loads a model from a JSON file, returning its weights and configuration.
///
/// # JSON Format (Human-Readable Model Storage)
/// ```json
/// {
///   "weights": [0.1, -0.5, 0.3, ...],
///   "config": {
///     "population_size": 128,
///     "mutation_rate": 0.05,
///     "activation": "Tanh",
///     "date_trained": "2024-01-15T10:30:00Z",
///     ...
///   }
/// }
/// ```
///
/// # Teaching Note: Model Persistence Design
/// This function separates model data from model behavior, following good software
/// architecture principles. The returned weights can be used to instantiate any
/// Individual type, while the config preserves the training hyperparameters for
/// reproducibility - a crucial aspect of scientific research.
///
/// # Error Handling Strategy
/// Uses Rust's `Result` type for robust error handling, providing detailed error
/// messages that help with debugging corrupted or incompatible model files.
///
/// # Returns
/// A `Result` containing a tuple of `(weights, config)`, or a descriptive error
/// if loading or deserialization fails.
pub fn load_model_from_file(
    path: &Path,
) -> Result<(Vec<f32>, Config), Box<dyn std::error::Error>> {
    // Read the entire JSON file into memory
    let json_content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read model file '{}': {}", path.display(), e))?;

    // Deserialize the JSON into our structured format
    let serializable: SerializableIndividual = serde_json::from_str(&json_content)
        .map_err(|e| format!("Failed to parse JSON in '{}': {}", path.display(), e))?;

    // Validate that the weights array has the expected size
    // This catches models trained with different network architectures
    if serializable.weights.len() != TOTAL_WEIGHTS {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Weight count mismatch in file '{}'. Expected {} weights, but found {}. \
                 This model may have been trained with a different network architecture.",
                path.display(),
                TOTAL_WEIGHTS,
                serializable.weights.len()
            ),
        )));
    }

    Ok((serializable.weights, serializable.config))
}
