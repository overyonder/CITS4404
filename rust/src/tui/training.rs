//! Training state, message passing, and backend logic for evolutionary training in the TUI.
use crate::config::Config;
use std::time::Instant;

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
    pub last_generation_time: Instant,
    pub generation_times: Vec<f32>, // time taken for each generation in seconds
    pub best_fitness_ever: f32,
    pub last_best_fitness: f32,
    pub improvement_times: Vec<Instant>, // when improvements occurred
}

impl TrainingState {
    /// Creates a new `TrainingState` with default values.
    #[allow(dead_code)]
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
            last_generation_time: now,
            generation_times: Vec::new(),
            best_fitness_ever: 0.0,
            last_best_fitness: 0.0,
            improvement_times: Vec::new(),
        }
    }
    
    /// Updates metrics when a generation completes
    pub fn update_generation_metrics(&mut self, new_best_fitness: f32, population_size: usize) {
        let now = Instant::now();
        let generation_duration = now.duration_since(self.last_generation_time).as_secs_f32();
        
        // Update generation timing
        self.generation_times.push(generation_duration);
        self.last_generation_time = now;
        
        // Calculate training rate (generations per second)
        let training_rate = if generation_duration > 0.0 { 1.0 / generation_duration } else { 0.0 };
        self.training_rate_history.push(training_rate);
        
        // Calculate total matches simulated (full round-robin tournament)
        let matches_this_generation = (population_size * (population_size - 1)) as u64;
        self.total_matches_simulated += matches_this_generation;
        
        // Track fitness improvements
        if new_best_fitness > self.best_fitness_ever {
            self.best_fitness_ever = new_best_fitness;
            self.improvement_times.push(now);
        }
        
        // Calculate improvement rate (fitness improvement per unit time)
        let total_time = now.duration_since(self.start_time).as_secs_f32();
        let improvement_rate = if total_time > 0.0 {
            (new_best_fitness - self.last_best_fitness) / generation_duration
        } else {
            0.0
        };
        self.improvement_rate_history.push(improvement_rate.max(0.0)); // Only positive improvements
        
        self.last_best_fitness = new_best_fitness;
    }
    
    /// Gets the current training rate (generations per second)
    pub fn get_current_training_rate(&self) -> f32 {
        self.training_rate_history.last().copied().unwrap_or(0.0)
    }
    
    /// Gets the current improvement rate (fitness improvement per unit time)
    pub fn get_current_improvement_rate(&self) -> f32 {
        self.improvement_rate_history.last().copied().unwrap_or(0.0)
    }
    
    /// Gets the total elapsed time in seconds
    pub fn get_elapsed_time(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }
    
    /// Calculates the maximum possible score based on game mechanics
    pub fn get_max_possible_score(&self) -> u32 {
        // Based on actual game constants from constants.rs and gamestate.rs:
        // - Game timeout: 2 * 16 * MAX_SCORE * WIDTH = 2 * 16 * 1 * 400 = 12,800 ticks
        // - Ball travels ~6.67 pixels/tick (WIDTH/TICK_RATE = 400/60)
        // - Paddle size: HEIGHT/8 = 37.5 pixels
        // - Court width: 400 pixels
        
        use crate::constants::{MAX_SCORE, WIDTH, HEIGHT, TICK_RATE};
        
        // Time limit per game in ticks
        let max_ticks = 2 * 16 * MAX_SCORE as u32 * WIDTH as u32;
        
        // Estimate minimum time for a complete rally (ball crossing court twice)
        // Ball speed ≈ WIDTH/TICK_RATE pixels per tick
        // Time to cross court ≈ WIDTH ticks / (WIDTH/TICK_RATE) = TICK_RATE ticks
        let min_rally_time = TICK_RATE as u32 * 2; // Round trip
        
        // Theoretical maximum rallies in a game
        let max_rallies = max_ticks / min_rally_time;
        
        // In CppEquivalent mode, score = returns + shots
        // Each rally can produce 1 return + potentially 1 shot per player
        // Conservative estimate: 60% of rallies result in successful returns/shots
        let max_returns_and_shots = (max_rallies as f32 * 0.6) as u32;
        
        // Add bonus for potential wins (in CppEquivalent this is secondary objective)
        let max_with_win_bonus = max_returns_and_shots + 1; // +1 for winning the game
        
        max_with_win_bonus.min(100) // Cap at reasonable maximum
    }
}

/// Messages sent from the training thread to the UI thread.
#[allow(dead_code)] // The UI doesn't handle all message types or fields yet.
pub enum TrainingMessage {
    /// Reports summary progress at the end of a generation.
    Progress {
        generation: usize,
        best_fitness: f32,
        avg_fitness: f32,
        worst_fitness: f32,
        genome_weights: Vec<f32>,
        total_matches_simulated: u64,
        training_rate: f32,
        improvement_rate: f32,
        max_possible_score: u32,
    },
    /// Indicates that the training process has finished normally.
    Finished,
    /// Indicates that training was stopped early due to convergence or stagnation.
    EarlyStopping {
        final_generation: usize,
        best_fitness: f32,
    },
}
