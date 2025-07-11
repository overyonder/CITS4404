//! C++ compatibility layer for loading trained models from legacy C++ application.
//!
//! # Teaching Note: Legacy Integration
//! This module demonstrates effective strategies for integrating with legacy systems:
//! - Robust parsing that handles multiple data formats
//! - Comprehensive error handling with informative messages
//! - Backwards compatibility while providing modern functionality
//! - Testing strategies for file format compatibility
//!
//! # Legacy Format Structure
//! The C++ fittest.log file uses a specific format:
//! - Header line: "4 8 16 4 1" (network architecture description)
//! - Generation markers: Single numbers on their own lines
//! - Individual data: "N weight1 weight2 ... weightN" where N is the count
//!
//! This format predates modern JSON serialization and requires careful parsing.

use crate::{
    config::{Config, Engine, FitnessFunc, MutationStrategy, ReproductionStrategy},
    constants::TOTAL_WEIGHTS,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tracing::{debug, warn};

/// Custom error types for C++ compatibility operations.
#[derive(Debug)]
pub enum CppCompatError {
    FileNotFound(String),
    WeightMismatch { expected: usize, found: usize },
    NoValidChampion,
    IoError(std::io::Error),
}

impl std::fmt::Display for CppCompatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CppCompatError::FileNotFound(path) => {
                write!(f, "C++ log file not found at: {}", path)
            }

            CppCompatError::WeightMismatch { expected, found } => {
                write!(f, "Weight count mismatch. Expected {}, found {}", expected, found)
            }
            CppCompatError::NoValidChampion => {
                write!(f, "No valid champion found in the C++ log file")
            }
            CppCompatError::IoError(e) => {
                write!(f, "I/O error reading C++ log file: {}", e)
            }
        }
    }
}

impl std::error::Error for CppCompatError {}

impl From<std::io::Error> for CppCompatError {
    fn from(error: std::io::Error) -> Self {
        CppCompatError::IoError(error)
    }
}

/// Metadata about the C++ log file structure and content.
/// 
/// # Teaching Note: Minimal Metadata Design
/// This struct contains only the essential information needed by the application.
/// Keeping metadata minimal reduces memory usage and simplifies the interface.
#[derive(Debug, Clone)]
pub struct CppLogMetadata {
    // Note: Most metadata fields have been removed as they were only used in tests
    // and don't contribute to the core functionality of loading champions.
}

/// Different format versions of C++ log files we can handle.
#[derive(Debug, Clone, PartialEq)]
pub enum CppLogFormat {
    /// Original format with simple generation markers
    Original,
    /// Extended format with additional metadata
    Extended,
    /// Unknown format that we'll attempt to parse anyway
    Unknown,
}

/// Enhanced C++ champion loader with comprehensive error handling and metadata extraction.
///
/// # Teaching Note: Robust File Parsing
/// This function demonstrates several important parsing principles:
/// - Defensive programming: Handle malformed data gracefully
/// - Progressive parsing: Extract what we can even if some data is invalid
/// - Rich error information: Help users understand what went wrong
/// - Metadata extraction: Provide insights into the loaded data
///
/// # Arguments
/// * `path` - Path to the C++ fittest.log file
///
/// # Returns
/// * `Ok((weights, config, metadata))` - Successfully loaded champion data
/// * `Err(CppCompatError)` - Detailed error information about what went wrong
pub fn load_cpp_champion_enhanced(
    path: &str,
) -> Result<(Vec<f32>, Config, CppLogMetadata), CppCompatError> {
    // Validate file existence with informative error
    let file_path = std::path::Path::new(path);
    if !file_path.exists() {
        return Err(CppCompatError::FileNotFound(format!(
            "{}\n\nMake sure the C++ training application has been run and produced a fittest.log file.",
            path
        )));
    }

    // Get file metadata (not used in minimal metadata)
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header line (not used in minimal metadata but needed for format detection)
    let header_line = lines.next().transpose()?;
    
    // Determine log format based on header structure
    let format_version = detect_log_format(&header_line);
    debug!("Detected C++ log format: {:?}", format_version);

    let mut last_champion_weights: Vec<f32> = Vec::new();
    let mut generation_count = 0u32;
    let mut parsing_errors = Vec::new();

    // Collect all lines for analysis
    let file_lines: Vec<String> = lines.collect::<Result<Vec<_>, _>>()?;
    debug!("Processing {} lines from C++ log file", file_lines.len());

    // Enhanced parsing with error tracking
    let mut _last_valid_weights_line = None;
    
    for (line_num, line) in file_lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        
        // Check if this is a generation marker (single number)
        if parts.len() == 1 {
            if let Ok(_gen) = parts[0].parse::<u32>() {
                generation_count += 1;
                debug!("Found generation marker {} at line {}", _gen, line_num + 1);
                continue;
            }
        }
        
        // Check if this is a weights line (first number is count, rest are weights)
        if parts.len() > 1 {
            if let Ok(weight_count) = parts[0].parse::<usize>() {
                let weight_strings = &parts[1..];
                
                // Attempt to parse all weights
                let mut weights = Vec::new();
                let mut parse_success = true;
                
                for (i, weight_str) in weight_strings.iter().enumerate() {
                    match weight_str.parse::<f32>() {
                        Ok(weight) => weights.push(weight),
                        Err(e) => {
                            parsing_errors.push(format!(
                                "Line {}: Invalid weight '{}' at position {}: {}",
                                line_num + 1, weight_str, i + 1, e
                            ));
                            parse_success = false;
                            break;
                        }
                    }
                }
                
                if parse_success {
                    if weight_count == weights.len() && weight_count == TOTAL_WEIGHTS {
                        _last_valid_weights_line = Some((line_num + 1, weights.clone()));
                        last_champion_weights = weights;
                        debug!("Found valid champion at line {} with {} weights", 
                               line_num + 1, weight_count);
                    } else if weight_count != weights.len() {
                        parsing_errors.push(format!(
                            "Line {}: Weight count mismatch. Header says {}, found {} weights",
                            line_num + 1, weight_count, weights.len()
                        ));
                    } else if weight_count != TOTAL_WEIGHTS {
                        parsing_errors.push(format!(
                            "Line {}: Unexpected weight count {}. Expected {}",
                            line_num + 1, weight_count, TOTAL_WEIGHTS
                        ));
                    }
                }
            }
        }
    }

    // Log parsing warnings if any
    if !parsing_errors.is_empty() {
        warn!("Encountered {} parsing issues in C++ log file:", parsing_errors.len());
        for (i, error) in parsing_errors.iter().enumerate().take(5) {
            warn!("  {}: {}", i + 1, error);
        }
        if parsing_errors.len() > 5 {
            warn!("  ... and {} more parsing issues", parsing_errors.len() - 5);
        }
    }

    // Validate that we found a valid champion
    if last_champion_weights.is_empty() {
        return Err(CppCompatError::NoValidChampion);
    }

    if last_champion_weights.len() != TOTAL_WEIGHTS {
        return Err(CppCompatError::WeightMismatch {
            expected: TOTAL_WEIGHTS,
            found: last_champion_weights.len(),
        });
    }

    // Create metadata about the loaded file
    let metadata = CppLogMetadata {};

    // Create a compatible configuration that matches C++ behavior
    let cpp_config = create_cpp_compatible_config(generation_count, &metadata);

    debug!("Successfully loaded C++ champion with {} weights from {} generations", 
           last_champion_weights.len(), generation_count);

    Ok((last_champion_weights, cpp_config, metadata))
}

/// Simplified interface that maintains backwards compatibility.
///
/// # Teaching Note: API Design
/// This function provides a simpler interface for existing code while internally
/// using the enhanced loader. This is a common pattern when improving APIs:
/// - Keep the old interface working to avoid breaking changes
/// - Add new, more powerful interfaces alongside
/// - Eventually deprecate the old interface when appropriate
pub fn load_cpp_champion(path: &str) -> Result<(Vec<f32>, Config), Box<dyn std::error::Error>> {
    let (weights, config, _metadata) = load_cpp_champion_enhanced(path)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    Ok((weights, config))
}

/// Detects the format version of a C++ log file based on its header.
fn detect_log_format(header: &Option<String>) -> CppLogFormat {
    match header {
        Some(header_line) => {
            let parts: Vec<&str> = header_line.trim().split_whitespace().collect();
            if parts.len() == 5 && parts.iter().all(|p| p.parse::<u32>().is_ok()) {
                CppLogFormat::Original
            } else if parts.len() > 5 {
                CppLogFormat::Extended
            } else {
                CppLogFormat::Unknown
            }
        }
        None => CppLogFormat::Unknown,
    }
}

/// Creates a configuration that closely matches C++ application behavior.
///
/// # Teaching Note: Legacy Compatibility
/// When integrating with legacy systems, it's important to match their behavior
/// as closely as possible. This includes:
/// - Using the same algorithmic choices (mutation strategies, fitness functions)
/// - Matching the same parameter defaults
/// - Preserving the same network architecture
fn create_cpp_compatible_config(generation_count: u32, _metadata: &CppLogMetadata) -> Config {
    let model_name = if generation_count > 0 {
        format!("C++ Champion ({} generations)", generation_count)
    } else {
        "C++ Champion (unknown generations)".to_string()
    };

    Config {
        name: Some(model_name),
        engine: Engine::Cpu, // C++ implementation is most similar to CPU engine
        generations: generation_count,
        
        // Match C++ application defaults exactly
        population_size: 128, // Common C++ default
        elite_count: 2,       // Typical C++ elite preservation
        mutation_rate: 0.05,
        mutation_strength: 0.1,
        
        // Use strategies that match C++ behavior
        reproduction_strategy: ReproductionStrategy::CppEquivalent,
        mutation_strategy: MutationStrategy::CppEquivalent,
        fitness_func: FitnessFunc::CppEquivalent,
        
        // C++ applications typically use simpler activation functions
        activation: crate::config::Activation::ClampedLinear,
        
        // Selection parameters (C++ compatibility)
        selection_strategy: crate::config::SelectionStrategy::Tournament,
        tournament_size: 3,     // Common tournament size
        truncation_rate: 0.3,   // Reasonable default
        crossover_rate: 0.8,    // Standard crossover rate
        
        // Adaptive mutation (disabled for C++ compatibility)
        adaptive_mutation: false,
        min_mutation_rate: 0.01,
        max_mutation_rate: 0.1,
        
        // Convergence criteria (conservative for compatibility)
        early_stopping_patience: None, // C++ didn't have early stopping
        fitness_threshold: None,
        track_diversity: false,        // C++ didn't track diversity
        normalize_fitness: false,      // C++ used raw fitness values
        
        // Other reasonable defaults for C++ compatibility
        random_ball_direction: false, // C++ version typically uses fixed direction
        concurrent: false,            // C++ version was single-threaded
        random_seed: None,           // C++ used system time as seed
        simulation_speed: 1.0,       // Normal speed for C++ compatibility
        champion_seed_path: None,    // C++ doesn't use champion seeding
        
        // Metadata
        date_trained: None, // Unknown from C++ logs
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rng, Rng};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper function to create test log files with known content.
    fn create_test_log_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    /// Helper to generate random weights for testing.
    fn generate_test_weights() -> Vec<f32> {
        let mut rng = rng();
        (0..TOTAL_WEIGHTS).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    #[test]
    fn test_load_valid_champion_original_format() {
        let weights = generate_test_weights();
        let weights_str = weights.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let content = format!("4 8 16 4 1\n1\n{} {}\n", TOTAL_WEIGHTS, weights_str);
        let file = create_test_log_file(&content);

        let result = load_cpp_champion_enhanced(file.path().to_str().unwrap());
        assert!(result.is_ok());
        
        let (loaded_weights, config, _metadata) = result.unwrap();
        assert_eq!(loaded_weights.len(), TOTAL_WEIGHTS);
        assert_eq!(config.engine, Engine::Cpu);
        assert!(config.name.unwrap().contains("C++ Champion"));
        // Metadata field checks removed since fields are no longer tracked
        
        // Compare weights with tolerance
        for (a, b) in loaded_weights.iter().zip(weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multiple_generations() {
        let weights1 = generate_test_weights();
        let weights2 = generate_test_weights();
        
        let weights1_str = weights1.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(" ");
        let weights2_str = weights2.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(" ");

        let content = format!(
            "4 8 16 4 1\n1\n{} {}\n2\n{} {}\n", 
            TOTAL_WEIGHTS, weights1_str,
            TOTAL_WEIGHTS, weights2_str
        );
        let file = create_test_log_file(&content);

        let result = load_cpp_champion_enhanced(file.path().to_str().unwrap());
        assert!(result.is_ok());
        
        let (loaded_weights, _, _metadata) = result.unwrap();
        // Metadata field checks removed since fields are no longer tracked
        
        // Should load the last (most recent) champion
        for (a, b) in loaded_weights.iter().zip(weights2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_malformed_data_handling() {
        let valid_weights = generate_test_weights();
        let valid_weights_str = valid_weights.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let content = format!(
            "4 8 16 4 1\n1\n10 a b c\n2\n{} {}\n3\n5 d e f\n",
            TOTAL_WEIGHTS, valid_weights_str
        );
        let file = create_test_log_file(&content);

        let result = load_cpp_champion_enhanced(file.path().to_str().unwrap());
        assert!(result.is_ok());
        
        let (loaded_weights, _, _metadata) = result.unwrap();
        assert_eq!(loaded_weights.len(), TOTAL_WEIGHTS);
        // Metadata field checks removed since fields are no longer tracked
        
        // Should have loaded the valid weights
        for (a, b) in loaded_weights.iter().zip(valid_weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weight_count_mismatch() {
        let content = "4 8 16 4 1\n1\n3 1.0 2.0 3.0\n"; // Wrong number of weights
        let file = create_test_log_file(content);

        let result = load_cpp_champion_enhanced(file.path().to_str().unwrap());
        assert!(result.is_err());
        
        match result.unwrap_err() {
            CppCompatError::NoValidChampion => {}, // Expected
            other => panic!("Expected NoValidChampion error, got: {}", other),
        }
    }

    #[test]
    fn test_file_not_found() {
        let result = load_cpp_champion_enhanced("non_existent_file.log");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            CppCompatError::FileNotFound(path) => {
                assert!(path.contains("non_existent_file.log"));
            }
            other => panic!("Expected FileNotFound error, got: {}", other),
        }
    }

    #[test]
    fn test_empty_file() {
        let file = create_test_log_file("");
        let result = load_cpp_champion_enhanced(file.path().to_str().unwrap());
        assert!(result.is_err());
        
        match result.unwrap_err() {
            CppCompatError::NoValidChampion => {}, // Expected
            other => panic!("Expected NoValidChampion error, got: {}", other),
        }
    }

    #[test]
    fn test_backwards_compatibility() {
        let weights = generate_test_weights();
        let weights_str = weights.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let content = format!("4 8 16 4 1\n1\n{} {}\n", TOTAL_WEIGHTS, weights_str);
        let file = create_test_log_file(&content);

        // Test that the old API still works
        let result = load_cpp_champion(file.path().to_str().unwrap());
        assert!(result.is_ok());
        
        let (loaded_weights, config) = result.unwrap();
        assert_eq!(loaded_weights.len(), TOTAL_WEIGHTS);
        assert_eq!(config.engine, Engine::Cpu);
    }


}
