//! Game world and physics constants for the Pong simulation.
//!
//! These constants define the fundamental properties of the Pong game environment,
//! including the playing field dimensions, paddle properties, ball physics, and
//! timing parameters.

// ----------------------------------------------------------------------------
// Game World Constants (Pong Environment)
// ----------------------------------------------------------------------------

/// The width of the Pong game area in pixels.
/// 
/// # Teaching Note: Coordinate System Design
/// This establishes the game's coordinate system. In Pong, the width is typically
/// wider than height to create the classic "tennis court" aspect ratio.
/// The 4:3 ratio (400:300) is chosen for:
/// - **Visual clarity**: Easy to see paddle and ball movements
/// - **Computational efficiency**: Nice round numbers for physics calculations
/// - **Historical accuracy**: Matches classic Pong proportions
pub const WIDTH: u16 = 400;

/// The height of the Pong game area in pixels.
///
/// # Teaching Note: Aspect Ratio Considerations
/// The height is intentionally smaller than width to create a landscape orientation.
/// This affects gameplay strategy - vertical movement is more constrained than
/// horizontal ball travel, making timing and positioning critical skills for AI.
pub const HEIGHT: u16 = 300;

/// The height of each paddle in pixels.
///
/// # Teaching Note: Game Balance Design
/// Paddle height (37.5px) is exactly 1/8th of the game height. This ratio determines
/// the difficulty of the game:
/// - **Too large**: Game becomes trivial, no challenge for AI
/// - **Too small**: Game becomes impossible, AI can't learn effectively
/// - **Just right**: Requires skill but allows successful defensive play
/// 
/// This value directly impacts the **fitness landscape** that the AI must learn to navigate.
pub const PADDLE_HEIGHT: f32 = HEIGHT as f32 / 8.0; // 37.5 pixels

/// The maximum score before a game ends.
///
/// # Teaching Note: Episode Length Design
/// Short games (max score = 1) create **dense learning signals**:
/// - Each point matters enormously for fitness evaluation
/// - Quick episode turnover allows more fitness evaluations per generation
/// - Reduces variance in fitness measurements
/// - Focuses learning on critical moments rather than endurance
pub const MAX_SCORE: u8 = 1;

// ----------------------------------------------------------------------------
// Physics and Simulation Constants
// ----------------------------------------------------------------------------

/// The number of simulation ticks that occur per second.
///
/// # Teaching Note: Temporal Discretization
/// This controls the simulation's temporal resolution. Higher values provide:
/// - **Better physics accuracy**: Smaller time steps reduce numerical integration errors
/// - **Smoother motion**: More position updates create fluid movement
/// - **Higher computational cost**: More calculations per simulated second
/// 
/// 60 FPS is chosen as the sweet spot for real-time feel without excessive computation.
pub const TICK_RATE: u16 = 60;

/// The maximum velocity of a paddle in pixels per tick.
///
/// # Teaching Note: Movement Constraints
/// This value (5.0 px/tick) determines how quickly paddles can react:
/// - **Too high**: Paddles become teleporters, game loses realism
/// - **Too low**: Paddles can't react to fast balls, game becomes impossible
/// - **Just right**: Requires prediction and positioning strategy
/// 
/// The formula `HEIGHT / TICK_RATE` ensures paddles can traverse the full height
/// in exactly 1 second, providing intuitive scaling.
pub const PADDLE_MAX_VEL: f32 = HEIGHT as f32 / TICK_RATE as f32; // 5.0 px/tick

/// The initial x-velocity of the ball in pixels per tick.
///
/// # Teaching Note: Ball Speed Tuning
/// Ball velocity (6.67 px/tick) determines game pace and difficulty:
/// - Crosses the field in ~60 ticks (1 second) for human-readable gameplay
/// - Fast enough to challenge AI reaction time
/// - Slow enough to allow learning and strategic positioning
pub const BALL_INITIAL_VEL_X: f32 = WIDTH as f32 / TICK_RATE as f32; // ~6.67 px/tick

/// The initial y-velocity of the ball in pixels per tick.
///
/// # Teaching Note: Symmetric Physics
/// Y-velocity matches X-velocity to create symmetric ball movement.
/// This ensures no inherent bias toward horizontal vs vertical strategies.
pub const BALL_INITIAL_VEL_Y: f32 = WIDTH as f32 / TICK_RATE as f32; // ~6.67 px/tick 