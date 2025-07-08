//! Handles the loading of trained models from files.

use crate::config::Config;
use crate::constants::TOTAL_WEIGHTS;
use bytemuck;
use std::io::Read;

/// Loads a model from a binary file, returning its weights and configuration.
///
/// This function is engine-agnostic. It reads the metadata and raw weights,
/// allowing the caller to decide which neural network engine to instantiate.
///
/// # File Format
/// 1. `u64` (little-endian): Length of the JSON config string.
/// 2. `[u8]`: The UTF-8 encoded JSON config string.
/// 3. `[f32]`: The raw `f32` weights.
///
/// # Returns
/// A `Result` containing a tuple of the `(weights, config)`,
/// or an error if reading or deserialization fails.
pub fn load_model_from_file(
    path: &std::path::Path,
) -> Result<(Vec<f32>, Config), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;

    // 1. Read config length
    let mut config_len_bytes = [0u8; 8];
    file.read_exact(&mut config_len_bytes)?;
    let config_len = u64::from_le_bytes(config_len_bytes);

    // 2. Read and deserialize config
    let mut config_bytes = vec![0u8; config_len as usize];
    file.read_exact(&mut config_bytes)?;
    let config: Config = serde_json::from_slice(&config_bytes)?;

    // 3. Read weights
    let mut weights_bytes = Vec::new();
    file.read_to_end(&mut weights_bytes)?;

    if weights_bytes.len() != TOTAL_WEIGHTS * std::mem::size_of::<f32>() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Weight data size mismatch in file '{}'. Expected {} bytes, but found {}. File may be corrupt or from an incompatible version.",
                path.display(),
                TOTAL_WEIGHTS * std::mem::size_of::<f32>(),
                weights_bytes.len()
            ),
        )));
    }

    let weights: Vec<f32> = bytemuck::cast_slice(&weights_bytes).to_vec();

    Ok((weights, config))
}
