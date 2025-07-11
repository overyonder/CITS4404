//! GPU Batch Round-Robin Evaluation Shader for Mass Parallel Genetic Algorithm Processing
//!
//! # Teaching Note: GPU Mass Parallelization
//! This shader demonstrates advanced GPU computing principles for genetic algorithms:
//! - **Batch Processing**: Evaluate entire populations simultaneously instead of individual-by-individual
//! - **Round-Robin Selection**: Full round-robin tournament exactly matching CPU version
//! - **Memory Coalescence**: Optimize memory access patterns for maximum GPU throughput
//! - **Workgroup Efficiency**: Use optimal workgroup sizes for different GPU architectures
//!
//! # Architecture Overview
//! This shader processes complete round-robin tournaments in parallel. Each GPU thread handles
//! one match between two individuals, with all matches running simultaneously.
//! This approach scales to thousands of matches running concurrently.

// Network architecture constants - MUST match constants.rs exactly
const INPUT_SIZE: u32 = 8u;
const HIDDEN1_SIZE: u32 = 16u;
const HIDDEN2_SIZE: u32 = 4u;
const OUTPUT_SIZE: u32 = 1u;

// Weight layout constants - MUST match the CPU-side layout
const L1_WEIGHTS: u32 = HIDDEN1_SIZE * (INPUT_SIZE + 1u);      // 16 * (8 + 1) = 144
const L2_WEIGHTS: u32 = HIDDEN2_SIZE * (HIDDEN1_SIZE + 1u);    // 4 * (16 + 1) = 68
const L3_WEIGHTS: u32 = OUTPUT_SIZE * (HIDDEN2_SIZE + 1u);     // 1 * (4 + 1) = 5
const TOTAL_WEIGHTS: u32 = L1_WEIGHTS + L2_WEIGHTS + L3_WEIGHTS; // 217

// Pong simulation constants - MUST match constants.rs exactly
const PADDLE_HEIGHT: f32 = 37.5;
const PADDLE_MAX_VEL: f32 = 5.0;
const WIDTH: f32 = 400.0;
const HEIGHT: f32 = 300.0;
const BALL_INITIAL_VEL_X: f32 = 6.666;
const BALL_INITIAL_VEL_Y: f32 = 6.666;
const MAX_SCORE: u32 = 1u;
const MAX_STEPS: u32 = 1800u;

/// GPU-optimized game state for parallel Pong simulation matching CPU version exactly
struct GameState {
    paddle1_pos: f32,     // Left paddle Y position
    paddle1_vel: f32,     // Left paddle Y velocity  
    paddle2_pos: f32,     // Right paddle Y position
    paddle2_vel: f32,     // Right paddle Y velocity
    ball_pos_x: f32,      // Ball X position
    ball_pos_y: f32,      // Ball Y position
    ball_vel_x: f32,      // Ball X velocity
    ball_vel_y: f32,      // Ball Y velocity
    score1: u32,          // Left player score
    score2: u32,          // Right player score
    returns1: u32,        // Left player returns
    returns2: u32,        // Right player returns
    shots1: u32,          // Left player shots
    shots2: u32,          // Right player shots
    steps: u32,           // Simulation steps taken
    game_over: u32,       // 0 = running, 1 = finished
}

/// Match result for round-robin evaluation
struct MatchResult {
    player1_primary: u32,   // Player 1 primary fitness
    player1_secondary: u32, // Player 1 secondary fitness  
    player2_primary: u32,   // Player 2 primary fitness
    player2_secondary: u32, // Player 2 secondary fitness
}

/// Configuration passed from CPU to GPU
struct BatchConfig {
    population_size: u32,     // Number of individuals in population
    total_matches: u32,       // Total matches in round-robin
    activation_type: u32,     // Activation function (0-5)
    random_seed: u32,         // Random seed for reproducibility
    fitness_function: u32,    // Fitness function type (0-2)
    workgroup_offset: u32,    // Offset for chunked dispatch
    random_ball_direction: u32, // Whether to randomize ball direction
}

// GPU Buffers - bound from Rust code
@group(0) @binding(0) var<storage, read> population_weights: array<f32>;     // All individual weights
@group(0) @binding(1) var<storage, read> match_assignments: array<u32>;     // Match pairs (i,j,i,j,...)
@group(0) @binding(2) var<storage, read_write> match_results: array<MatchResult>; // Results output
@group(0) @binding(3) var<uniform> config: BatchConfig;                     // Configuration

/// High-performance activation functions matching CPU version exactly
fn activate(x: f32, activation_type: u32) -> f32 {
    switch activation_type {
        case 0u: { return clamp(x, -1.0, 1.0); }              // ClampedLinear
        case 1u: { return tanh(x); }                          // Tanh
        case 2u: { return max(0.0, x); }                      // ReLU
        case 3u: { return atan(x); }                          // Atan
        case 4u: { return x; }                                // Linear
        case 5u: { return 1.0 / (1.0 + exp(-x)); }          // Sigmoid
        default: { return x; }                                // Fallback to linear
    }
}

/// GPU-optimized neural network forward propagation matching CPU version exactly
fn forward_propagate_individual(weights_offset: u32, input: array<f32, INPUT_SIZE>, activation_type: u32) -> f32 {
    var offset = weights_offset;
    
    // Layer 1: Input -> Hidden1 (8 -> 16)
    var hidden1: array<f32, HIDDEN1_SIZE>;
    for (var i = 0u; i < HIDDEN1_SIZE; i = i + 1u) {
        let neuron_weight_offset = offset + i * (INPUT_SIZE + 1u);
        var sum = 0.0;
        for (var j = 0u; j < INPUT_SIZE; j = j + 1u) {
            sum = sum + input[j] * population_weights[neuron_weight_offset + j];
        }
        // Add bias
        sum = sum + population_weights[neuron_weight_offset + INPUT_SIZE];
        hidden1[i] = activate(sum, activation_type);
    }
    offset = offset + L1_WEIGHTS;
    
    // Layer 2: Hidden1 -> Hidden2 (16 -> 4)
    var hidden2: array<f32, HIDDEN2_SIZE>;
    for (var i = 0u; i < HIDDEN2_SIZE; i = i + 1u) {
        let neuron_weight_offset = offset + i * (HIDDEN1_SIZE + 1u);
        var sum = 0.0;
        for (var j = 0u; j < HIDDEN1_SIZE; j = j + 1u) {
            sum = sum + hidden1[j] * population_weights[neuron_weight_offset + j];
        }
        // Add bias
        sum = sum + population_weights[neuron_weight_offset + HIDDEN1_SIZE];
        hidden2[i] = activate(sum, activation_type);
    }
    offset = offset + L2_WEIGHTS;

    // Layer 3: Hidden2 -> Output (4 -> 1)
    var output_sum = 0.0;
    for (var j = 0u; j < HIDDEN2_SIZE; j = j + 1u) {
        output_sum = output_sum + hidden2[j] * population_weights[offset + j];
    }
    // Add bias
    output_sum = output_sum + population_weights[offset + HIDDEN2_SIZE];

    // Final output is not activated, consistent with CPU version
    return output_sum;
}

/// Fast pseudo-random number generator matching CPU behavior
var<private> rng_state: u32;

fn init_rng(seed: u32, thread_id: u32) {
    rng_state = seed ^ (thread_id * 747796405u + 2891336453u);
}

fn next_random() -> u32 {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

fn random_f32() -> f32 {
    return f32(next_random()) / 4294967296.0; // 2^32
}

fn random_range(min_val: f32, max_val: f32) -> f32 {
    return min_val + random_f32() * (max_val - min_val);
}

/// Gaussian random number generator for paddle deflection (Box-Muller transform)
var<private> gauss_cached: bool = false;
var<private> gauss_cache: f32 = 0.0;

fn gaussian_random(mean: f32, std_dev: f32) -> f32 {
    if (gauss_cached) {
        gauss_cached = false;
        return gauss_cache * std_dev + mean;
    }
    
    let u1 = random_f32();
    let u2 = random_f32();
    
    let mag = std_dev * sqrt(-2.0 * log(u1));
    gauss_cache = mag * cos(6.28318530718 * u2);
    gauss_cached = true;
    
    return mag * sin(6.28318530718 * u2) + mean;
}

/// High-performance Pong game simulation exactly matching CPU version
fn simulate_pong_match(individual1_offset: u32, individual2_offset: u32, activation_type: u32, fitness_func: u32) -> MatchResult {
    var state = GameState();
    
    // Initialize game state exactly like CPU version
    state.paddle1_pos = HEIGHT / 2.0;
    state.paddle1_vel = 0.0;
    state.paddle2_pos = HEIGHT / 2.0;
    state.paddle2_vel = 0.0;
    state.ball_pos_x = WIDTH / 2.0;
    state.ball_pos_y = HEIGHT / 2.0;
    state.ball_vel_x = BALL_INITIAL_VEL_X;
    state.ball_vel_y = BALL_INITIAL_VEL_Y;
    
    // Randomize ball direction if configured
    if (config.random_ball_direction != 0u) {
        if (random_f32() < 0.5) {
            state.ball_vel_x = -state.ball_vel_x;
        }
        state.ball_vel_y = random_range(-BALL_INITIAL_VEL_Y, BALL_INITIAL_VEL_Y);
    }
    
    state.score1 = 0u;
    state.score2 = 0u;
    state.returns1 = 0u;
    state.returns2 = 0u;
    state.shots1 = 0u;
    state.shots2 = 0u;
    state.steps = 0u;
    state.game_over = 0u;
    
    // Main game loop - exactly matching CPU version logic
    let timelimit = 2u * 16u * MAX_SCORE * u32(WIDTH); // ~25,600 ticks like CPU
    
    while (state.game_over == 0u && state.steps < timelimit) {
        state.steps++;
        
        // Create neural network input exactly matching CPU normalization
        var nn_input1: array<f32, INPUT_SIZE>;
        var nn_input2: array<f32, INPUT_SIZE>;
        
        // Player 1 input (left paddle perspective)
        nn_input1[0] = state.paddle1_pos / HEIGHT;                    
        nn_input1[1] = state.paddle2_pos / HEIGHT;                    
        nn_input1[2] = state.ball_pos_x / WIDTH;                      
        nn_input1[3] = state.ball_pos_y / HEIGHT;                     
        nn_input1[4] = state.ball_vel_x / 10.0;                       
        nn_input1[5] = state.ball_vel_y / 10.0;                       
        nn_input1[6] = f32(state.score1) / f32(MAX_SCORE);           
        nn_input1[7] = f32(state.score2) / f32(MAX_SCORE);           

        // Player 2 input (right paddle perspective, mirrored)
        nn_input2[0] = state.paddle2_pos / HEIGHT;                    
        nn_input2[1] = state.paddle1_pos / HEIGHT;                    
        nn_input2[2] = (WIDTH - state.ball_pos_x) / WIDTH;           
        nn_input2[3] = state.ball_pos_y / HEIGHT;                     
        nn_input2[4] = -state.ball_vel_x / 10.0;                      
        nn_input2[5] = state.ball_vel_y / 10.0;                       
        nn_input2[6] = f32(state.score2) / f32(MAX_SCORE);           
        nn_input2[7] = f32(state.score1) / f32(MAX_SCORE);           
        
        // Get paddle movements from neural networks
        let paddle1_output = forward_propagate_individual(individual1_offset, nn_input1, activation_type);
        let paddle2_output = forward_propagate_individual(individual2_offset, nn_input2, activation_type);
        
        // Update paddle velocities and positions exactly like CPU
        state.paddle1_vel = clamp(paddle1_output * PADDLE_MAX_VEL, -PADDLE_MAX_VEL, PADDLE_MAX_VEL);
        state.paddle2_vel = clamp(paddle2_output * PADDLE_MAX_VEL, -PADDLE_MAX_VEL, PADDLE_MAX_VEL);
        
        state.paddle1_pos = clamp(state.paddle1_pos + state.paddle1_vel, 
                                 PADDLE_HEIGHT / 2.0, HEIGHT - PADDLE_HEIGHT / 2.0);
        state.paddle2_pos = clamp(state.paddle2_pos + state.paddle2_vel,
                                 PADDLE_HEIGHT / 2.0, HEIGHT - PADDLE_HEIGHT / 2.0);
        
        // Update ball position
        state.ball_pos_x += state.ball_vel_x;
        state.ball_pos_y += state.ball_vel_y;
        
        // Ball collision with top/bottom walls
        if (state.ball_pos_y <= 0.0 || state.ball_pos_y >= HEIGHT) {
            state.ball_vel_y = -state.ball_vel_y;
            state.ball_pos_y = clamp(state.ball_pos_y, 0.0, HEIGHT);
        }
        
        var paddle_hit = false;
        
        // Left paddle collision detection exactly matching CPU
        if (state.ball_vel_x < 0.0 && state.ball_pos_x <= 10.0) {
            let paddle_top = state.paddle1_pos - PADDLE_HEIGHT / 2.0;
            let paddle_bottom = state.paddle1_pos + PADDLE_HEIGHT / 2.0;
            
            if (state.ball_pos_y >= paddle_top && state.ball_pos_y <= paddle_bottom) {
                state.ball_pos_x = 10.0; // Prevent penetration
                paddle_hit = true;
                
                // Physics response with momentum transfer like CPU
                state.ball_vel_x = abs(state.ball_vel_x); // Reverse direction
                state.ball_vel_y += state.paddle1_vel;    // Transfer paddle momentum
                
                // Add Gaussian random deflection like CPU  
                state.ball_vel_y += gaussian_random(0.0, 0.05) * BALL_INITIAL_VEL_Y;
                
                state.returns1++;
            }
        }
        // Right paddle collision detection exactly matching CPU  
        else if (state.ball_vel_x > 0.0 && state.ball_pos_x >= WIDTH - 10.0) {
            let paddle_top = state.paddle2_pos - PADDLE_HEIGHT / 2.0;
            let paddle_bottom = state.paddle2_pos + PADDLE_HEIGHT / 2.0;
            
            if (state.ball_pos_y >= paddle_top && state.ball_pos_y <= paddle_bottom) {
                state.ball_pos_x = WIDTH - 10.0; // Prevent penetration
                paddle_hit = true;
                
                // Physics response with momentum transfer like CPU
                state.ball_vel_x = -abs(state.ball_vel_x); // Reverse direction  
                state.ball_vel_y += state.paddle2_vel;     // Transfer paddle momentum
                
                // Add Gaussian random deflection like CPU
                state.ball_vel_y += gaussian_random(0.0, 0.05) * BALL_INITIAL_VEL_Y;
                
                state.returns2++;
            }
        }
        
        // Shot detection exactly matching CPU predictive analysis
        if (state.ball_vel_x < 0.0 && abs(state.ball_vel_x) > 0.001) {
            // Ball moving toward left player
            let time_to_wall = -state.ball_pos_x / state.ball_vel_x;
            let shot_y = state.ball_pos_y + state.ball_vel_y * time_to_wall;
            
            if (shot_y >= 0.0 && shot_y <= HEIGHT) {
                let paddle_top = state.paddle1_pos - PADDLE_HEIGHT / 2.0;
                let paddle_bottom = state.paddle1_pos + PADDLE_HEIGHT / 2.0;
                
                if (shot_y < paddle_top || shot_y > paddle_bottom) {
                    state.shots2++; // Right player executed a shot
                }
            }
        } else if (state.ball_vel_x > 0.0 && abs(state.ball_vel_x) > 0.001) {
            // Ball moving toward right player
            let time_to_wall = (WIDTH - state.ball_pos_x) / state.ball_vel_x;
            let shot_y = state.ball_pos_y + state.ball_vel_y * time_to_wall;
            
            if (shot_y >= 0.0 && shot_y <= HEIGHT) {
                let paddle_top = state.paddle2_pos - PADDLE_HEIGHT / 2.0;
                let paddle_bottom = state.paddle2_pos + PADDLE_HEIGHT / 2.0;
                
                if (shot_y < paddle_top || shot_y > paddle_bottom) {
                    state.shots1++; // Left player executed a shot
                }
            }
        }
        
        // Scoring detection and game state reset exactly like CPU
        if (!paddle_hit) {
            if (state.ball_pos_x < 0.0) {
                // Right player scores
                state.score2++;
                state.paddle1_pos = HEIGHT / 2.0;
                state.paddle2_pos = HEIGHT / 2.0;
                
                // Reset ball like CPU
                state.ball_pos_x = WIDTH / 2.0;
                state.ball_pos_y = HEIGHT / 2.0;
                state.ball_vel_x = BALL_INITIAL_VEL_X; // Serve to left
                if (config.random_ball_direction != 0u) {
                    state.ball_vel_y = random_range(-BALL_INITIAL_VEL_Y, BALL_INITIAL_VEL_Y);
                } else {
                    state.ball_vel_y = BALL_INITIAL_VEL_Y;
                }
                
                if (state.score2 >= MAX_SCORE) {
                    state.game_over = 1u;
                }
            } else if (state.ball_pos_x > WIDTH) {
                // Left player scores
                state.score1++;
                state.paddle1_pos = HEIGHT / 2.0;
                state.paddle2_pos = HEIGHT / 2.0;
                
                // Reset ball like CPU
                state.ball_pos_x = WIDTH / 2.0;
                state.ball_pos_y = HEIGHT / 2.0;
                state.ball_vel_x = -BALL_INITIAL_VEL_X; // Serve to right
                if (config.random_ball_direction != 0u) {
                    state.ball_vel_y = random_range(-BALL_INITIAL_VEL_Y, BALL_INITIAL_VEL_Y);
                } else {
                    state.ball_vel_y = BALL_INITIAL_VEL_Y;
                }
                
                if (state.score1 >= MAX_SCORE) {
                    state.game_over = 1u;
                }
            }
        }
    }
    
    // Calculate fitness exactly matching CPU version
    var result = MatchResult();
    
    switch fitness_func {
        case 0u: { // CppEquivalent - exactly like CPU
            let left_wins = select(0u, 1u, state.score1 > state.score2);
            let right_wins = select(0u, 1u, state.score2 > state.score1);
            let left_primary = state.returns1 + state.shots1;
            let right_primary = state.returns2 + state.shots2;
            
            result.player1_primary = left_primary;
            result.player1_secondary = left_wins;
            result.player2_primary = right_primary;
            result.player2_secondary = right_wins;
        }
        case 1u: { // ReturnFocused - exactly like CPU
            var left_score = state.returns1;
            var right_score = state.returns2;
            
            if (state.score1 > state.score2) {
                left_score += 5u;
            } else if (state.score2 > state.score1) {
                right_score += 5u;
            }
            
            result.player1_primary = left_score;
            result.player1_secondary = 0u;
            result.player2_primary = right_score;
            result.player2_secondary = 0u;
        }
        case 2u: { // VictoryOptimized - exactly like CPU
            var left_score = state.returns1;
            var right_score = state.returns2;
            
            if (state.score1 >= MAX_SCORE) {
                left_score += 10u;
            }
            if (state.score2 >= MAX_SCORE) {
                right_score += 10u;
            }
            
            result.player1_primary = left_score;
            result.player1_secondary = 0u;
            result.player2_primary = right_score;
            result.player2_secondary = 0u;
        }
        default: { // Fallback
            result.player1_primary = state.returns1;
            result.player1_secondary = 0u;
            result.player2_primary = state.returns2;
            result.player2_secondary = 0u;
        }
    }
    
    return result;
}

/// Main compute shader entry point for round-robin evaluation
///
/// Each GPU thread processes one match in the round-robin tournament.
/// For N individuals, there are N*(N-1) total matches.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let match_idx = config.workgroup_offset + global_id.x;

    // Boundary check
    if (match_idx >= config.total_matches) {
        return;
    }
    
    // Get the two players for this match
    let player1_idx = match_assignments[match_idx * 2u];
    let player2_idx = match_assignments[match_idx * 2u + 1u];
    
    // Initialize RNG for this match
    init_rng(config.random_seed, match_idx);
    
    // Calculate weight offsets
    let player1_offset = player1_idx * TOTAL_WEIGHTS;
    let player2_offset = player2_idx * TOTAL_WEIGHTS;
    
    // Run the match
    let result = simulate_pong_match(player1_offset, player2_offset, config.activation_type, config.fitness_function);
    
    // Store results
    match_results[match_idx] = result;
} 