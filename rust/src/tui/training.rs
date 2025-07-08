//! Training state, message passing, and backend logic for evolutionary training in the TUI.
use crate::config::Config;
use std::time::{Duration, Instant};

/// Holds the state for the training view.
pub struct TrainingState {
    pub running: bool,
    pub start_time: Instant,
    pub current_generation: usize,
    pub total_generations: usize,
    pub best_fitness: f32,
    pub fitness_history: Vec<f32>,
    pub genome_weights: Vec<f32>,
    
    // New metrics for enhanced tracking
    pub total_matches_simulated: u64,
    pub training_rate_history: Vec<f32>, // generations per second over time
    pub improvement_rate_history: Vec<f32>, // fitness improvement per unit time
    
    // For improvement rate averaging
    improvement_rate_samples: Vec<(Instant, f32)>, // (timestamp, improvement_rate)
}

impl TrainingState {
    /// Creates a new `TrainingState` with default values.
    pub fn new(config: &Config) -> Self {
        let now = Instant::now();
        Self {
            running: true,
            start_time: now,
            current_generation: 0,
            total_generations: config.generations as usize,
            best_fitness: 0.0,
            fitness_history: Vec::new(),
            genome_weights: Vec::new(),
            
            // Initialize new metrics
            total_matches_simulated: 0,
            training_rate_history: Vec::new(),
            improvement_rate_history: Vec::new(),
            
            // Initialize improvement rate samples
            improvement_rate_samples: Vec::new(),
        }
    }
    

    
    /// Gets the current training rate (generations per second)
    pub fn get_current_training_rate(&self) -> f32 {
        self.training_rate_history.last().copied().unwrap_or(0.0)
    }
    
    /// Adds a new improvement rate sample with current timestamp
    pub fn add_improvement_rate_sample(&mut self, improvement_rate: f32) {
        let now = Instant::now();
        self.improvement_rate_samples.push((now, improvement_rate));
        
        // Keep only samples from the last 15 seconds (with some buffer)
        let cutoff_time = now - Duration::from_secs(15);
        self.improvement_rate_samples.retain(|(timestamp, _)| *timestamp > cutoff_time);
        
        // Also add to history for display
        self.improvement_rate_history.push(improvement_rate);
    }
    
    /// Gets the current improvement rate (average over last 10 seconds)
    pub fn get_current_improvement_rate(&self) -> f32 {
        if self.improvement_rate_samples.is_empty() {
            return 0.0;
        }
        
        let now = Instant::now();
        let cutoff_time = now - Duration::from_secs(10);
        
        // Get samples from the last 10 seconds
        let recent_samples: Vec<f32> = self.improvement_rate_samples
            .iter()
            .filter(|(timestamp, _)| *timestamp > cutoff_time)
            .map(|(_, rate)| *rate)
            .collect();
        
        if recent_samples.is_empty() {
            // If no recent samples, fall back to the most recent sample
            self.improvement_rate_samples.last().map(|(_, rate)| *rate).unwrap_or(0.0)
        } else {
            // Calculate average of recent samples
            recent_samples.iter().sum::<f32>() / recent_samples.len() as f32
        }
    }
    
}

/// Messages sent from the training thread to the UI thread.
pub enum TrainingMessage {
    /// Reports summary progress at the end of a generation.
    Progress {
        generation: usize,
        best_fitness: f32,
        genome_weights: Vec<f32>,
        total_matches_simulated: u64,
        training_rate: f32,
        improvement_rate: f32,
    },
    /// Indicates that the training process has finished normally.
    Finished,
    /// Indicates that training was stopped early due to convergence or stagnation.
    EarlyStopping {
        final_generation: usize,
        best_fitness: f32,
    },
}
