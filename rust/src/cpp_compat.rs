//! C++ compatibility layer for loading trained models.

use crate::constants::TOTAL_WEIGHTS;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Loads the weights of the fittest individual from a C++ `fittest.log` file.
///
/// The function reads the entire log file to find the last generation's data
/// and returns the weights of the first individual listed in that generation.
pub fn load_cpp_champion(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Check if the file exists first, to provide a clearer error message.
    if !std::path::Path::new(path).exists() {
        return Err(format!("Log file not found at: {}", path).into());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Skip header
    lines.next();

    let mut last_champion_weights: Vec<f32> = Vec::new();

    let file_lines: Vec<String> = lines.collect::<Result<_,_>>()?;

    // Find the last generation
    if let Some(last_line_with_weights) = file_lines.iter().rev().find(|l| !l.trim().is_empty() && l.split_whitespace().count() > 1) {
        let weights: Vec<f32> = last_line_with_weights
            .split_whitespace()
            .skip(1) // The first element is the number of weights, so we skip it.
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

        if weights.len() == TOTAL_WEIGHTS {
            last_champion_weights = weights;
        } else {
            return Err(format!(
                "Weight count mismatch. Expected {}, found {}.",
                TOTAL_WEIGHTS,
                weights.len()
            )
            .into());
        }
    }

    if last_champion_weights.is_empty() {
        Err("No valid champion found in the log file.".into())
    } else {
        Ok(last_champion_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rng, Rng};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_valid_champion() {
        let mut file = NamedTempFile::new().unwrap();
        let mut rng = rng();
        let weights: Vec<f32> = (0..217).map(|_| rng.random_range(-1.0..1.0)).collect();
        let weights_str: String = weights
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let content = format!("4 8 16 4 1\n1\n217 {}\n", weights_str);
        file.write_all(content.as_bytes()).unwrap();

        let result = load_cpp_champion(file.path().to_str().unwrap());
        assert!(result.is_ok());
        let loaded_weights = result.unwrap();
        assert_eq!(loaded_weights.len(), 217);
        // Compare floats with a tolerance
        for (a, b) in loaded_weights.iter().zip(weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_malformed_data_is_skipped() {
        let mut file = NamedTempFile::new().unwrap();
        let mut rng = rng();
        let valid_weights: Vec<f32> = (0..217).map(|_| rng.random_range(-1.0..1.0)).collect();
        let valid_weights_str: String = valid_weights
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let content = format!(
            "4 8 16 4 1\n1\n10 a b c\n1\n217 {}\n1\n5 d e f\n",
            valid_weights_str
        );
        file.write_all(content.as_bytes()).unwrap();

        let result = load_cpp_champion(file.path().to_str().unwrap());
        assert!(result.is_ok());
        let loaded_weights = result.unwrap();

        // It should have loaded the first valid genome it found
        assert_eq!(loaded_weights.len(), 217);
        for (a, b) in loaded_weights.iter().zip(valid_weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_incorrect_weight_count() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Generation 1:").unwrap();
        writeln!(file, "1.0 2.0 3.0").unwrap(); // Not enough weights

        let result = load_cpp_champion(file.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Weight count mismatch"));
    }

    #[test]
    fn test_file_not_found() {
        let result = load_cpp_champion("non_existent_file.log");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Log file not found"));
    }

    #[test]
    fn test_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let result = load_cpp_champion(file.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No valid champion found"));
    }
}
