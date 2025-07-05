//! C++ compatibility layer for loading trained models.

use crate::constants::TOTAL_WEIGHTS;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Loads the weights of the fittest individual from a C++ `fittest.log` file.
///
/// The function reads the entire log file to find the last generation's data
/// and returns the weights of the first individual listed in that generation.
pub fn load_cpp_champion(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Skip header
    lines.next();

    let mut last_champion_weights: Vec<f32> = Vec::new();

    let file_lines: Vec<String> = lines.collect::<Result<_,_>>()?;

    // Find the last generation
    if let Some(last_line_with_weights) = file_lines.iter().rev().find(|l| !l.trim().is_empty() && l.split_whitespace().count() > 1) {
        let weights_str: Vec<&str> = last_line_with_weights.split_whitespace().collect();
        // The first element is the number of weights, so we skip it.
        let weights: Vec<f32> = weights_str[1..].iter().map(|s| s.parse::<f64>().unwrap() as f32).collect();
        if weights.len() == TOTAL_WEIGHTS {
            last_champion_weights = weights;
        } else {
            return Err("Weight count mismatch between C++ log and Rust configuration.".into());
        }
    }

    if last_champion_weights.is_empty() {
        Err("No valid champion found in the log file.".into())
    } else {
        Ok(last_champion_weights)
    }
}
