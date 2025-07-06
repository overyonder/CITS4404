//! Logic for discovering and loading simulation models from the filesystem.

use crate::config::Config;
use crate::tui::app::ModelInfo;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Scans a directory for model files and parses their metadata.
///
/// This function iterates through all files in the given directory, attempts to
/// parse the `Config` metadata from the header of each file, and returns a
/// vector of `ModelInfo` structs for valid models.
pub fn load_models_from_dir(dir: &Path) -> io::Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Ok(config) = load_config_from_file(&path) {
                models.push(ModelInfo {
                    path,
                    config,
                });
            }
        }
    }
    Ok(models)
}

/// Reads the `Config` from the header of a saved model file.
///
/// The model file is expected to be structured with a `u64` length prefix
/// for the JSON config, followed by the JSON data itself.
fn load_config_from_file(path: &Path) -> io::Result<Config> {
    let mut file = fs::File::open(path)?;

    // 1. Read the length of the config JSON
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes);

    // 2. Read the config JSON and deserialize it
    let mut config_bytes = vec![0; len as usize];
    file.read_exact(&mut config_bytes)?;
    let config: Config = serde_json::from_slice(&config_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(config)
}

/// Loads the raw f32 weights from a saved model file, skipping the config header.
pub fn load_weights_from_file(path: &Path) -> io::Result<Vec<f32>> {
    let mut file = fs::File::open(path)?;

    // 1. Read the length of the config JSON and skip it
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes);
    file.seek(SeekFrom::Current(len as i64))?;

    // 2. Read the rest of the file for the weights
    let mut weights_bytes = Vec::new();
    file.read_to_end(&mut weights_bytes)?;

    // 3. Convert bytes to Vec<f32>
    let weights: Vec<f32> = weights_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    Ok(weights)
}
